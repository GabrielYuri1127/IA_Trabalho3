"""Trabalho 3 FIA - CLEVR simplificado com raciocinio neuro-simbolico.

Rode com:
    pip install torch numpy matplotlib LTNtorch
    python clevr_ltn_experimentos.py --runs 5 --epochs 350 --out resultados_clevr_ltn.csv --plot-dir figuras

O codigo gera 5 datasets aleatorios com 25 objetos, treina predicados fuzzy
para atributos e relacoes espaciais, calcula satisfatibilidade das formulas
pedidas no enunciado e reporta acuracia, precisao, recall e F1.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    import ltn
    HAS_LTN = True
except ModuleNotFoundError:
    ltn = None
    HAS_LTN = False

COLORS = ["red", "green", "blue"]
SHAPES = ["circle", "square", "cylinder", "cone", "triangle"]
EPS = 1e-7


@dataclass
class Scene:
    x: torch.Tensor
    color: torch.Tensor
    shape: torch.Tensor
    size: torch.Tensor
    seed: int

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot(idx: np.ndarray, depth: int) -> np.ndarray:
    y = np.zeros((len(idx), depth), dtype=np.float32)
    y[np.arange(len(idx)), idx] = 1.0
    return y


def generate_scene(seed: int, n: int = 25) -> Scene:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.03, 0.97, size=(n, 2)).astype(np.float32)
    color = rng.integers(0, 3, size=n)
    shape = rng.integers(0, 5, size=n)
    size = rng.integers(0, 2, size=n)
    x = np.concatenate([pos, one_hot(color, 3), one_hot(shape, 5), size[:, None]], axis=1)
    return Scene(torch.tensor(x).float(), torch.tensor(color), torch.tensor(shape), torch.tensor(size), seed)


def pair_inputs(x: torch.Tensor):
    n, d = x.shape
    a = x[:, None, :].expand(n, n, d).reshape(n * n, d)
    b = x[None, :, :].expand(n, n, d).reshape(n * n, d)
    return a, b


def triple_inputs(x: torch.Tensor):
    n, d = x.shape
    a = x[:, None, None, :].expand(n, n, n, d).reshape(n ** 3, d)
    b = x[None, :, None, :].expand(n, n, n, d).reshape(n ** 3, d)
    c = x[None, None, :, :].expand(n, n, n, d).reshape(n ** 3, d)
    return a, b, c


def gt_left(s: Scene, margin: float = 0.02):
    return (s.x[:, 0, None] + margin < s.x[None, :, 0]).float()


def gt_right(s: Scene):
    return gt_left(s).T


def gt_below(s: Scene, margin: float = 0.02):
    return (s.x[:, 1, None] + margin < s.x[None, :, 1]).float()


def gt_above(s: Scene):
    return gt_below(s).T


def gt_close(s: Scene, threshold: float = 0.25):
    dist = torch.cdist(s.x[:, :2], s.x[:, :2])
    return ((dist < threshold) & (dist > 0)).float()


def gt_close_soft(s: Scene):
    dist2 = torch.cdist(s.x[:, :2], s.x[:, :2]) ** 2
    y = torch.exp(-2.0 * dist2)
    y.fill_diagonal_(0.0)
    return y


def gt_same_size(s: Scene):
    return (s.size[:, None] == s.size[None, :]).float()


def gt_can_stack(s: Scene):
    support_shape = s.shape[None, :]
    support_ok = ~((support_shape == SHAPES.index("cone")) | (support_shape == SHAPES.index("triangle")))
    same_size = s.size[:, None] == s.size[None, :]
    aligned = torch.abs(s.x[:, None, 0] - s.x[None, :, 0]) < 0.15
    return (support_ok & (same_size | aligned)).float()


def gt_between(s: Scene):
    left, right, n = gt_left(s), gt_right(s), s.n
    y = torch.zeros(n, n, n)
    for x in range(n):
        for a in range(n):
            for b in range(n):
                if len({x, a, b}) == 3:
                    y[x, a, b] = float((left[a, x] and right[b, x]) or (left[b, x] and right[a, x]))
    return y


class Unary(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(11, 24), nn.ELU(), nn.Linear(24, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(-1).clamp(EPS, 1 - EPS)


class Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(22, 32), nn.ELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=-1)).squeeze(-1).clamp(EPS, 1 - EPS)


class Ternary(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(33, 40), nn.ELU(), nn.Linear(40, 1), nn.Sigmoid())

    def forward(self, a, b, c):
        return self.net(torch.cat([a, b, c], dim=-1)).squeeze(-1).clamp(EPS, 1 - EPS)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.color = nn.ModuleDict({c: Unary() for c in COLORS})
        self.shape = nn.ModuleDict({f: Unary() for f in SHAPES})
        self.size = nn.ModuleDict({"small": Unary(), "big": Unary()})
        rels = ["left", "right", "below", "above", "close", "same_size", "can_stack"]
        self.rel = nn.ModuleDict({r: Binary() for r in rels})
        self.between = Ternary()

    def unary(self, x):
        out = {f"is_{k}": p(x) for k, p in self.color.items()}
        out.update({f"is_{k}": p(x) for k, p in self.shape.items()})
        out.update({f"is_{k}": p(x) for k, p in self.size.items()})
        return out

    def binary(self, s: Scene):
        a, b = pair_inputs(s.x)
        return {k: p(a, b).reshape(s.n, s.n) for k, p in self.rel.items()}

    def ternary(self, s: Scene):
        a, b, c = triple_inputs(s.x)
        return self.between(a, b, c).reshape(s.n, s.n, s.n)


def l_not(x): return 1 - x

def l_and(*xs):
    y = xs[0]
    for x in xs[1:]: y = y * x
    return y

def l_or(*xs):
    y = 1 - xs[0]
    for x in xs[1:]: y = y * (1 - x)
    return 1 - y

def l_imp(p, q): return (1 - p + p * q).clamp(EPS, 1)

def l_eq(p, q): return l_and(l_imp(p, q), l_imp(q, p))

def forall(x, p=2): return 1 - torch.mean((1 - x).clamp_min(0) ** p) ** (1 / p)

def exists(x, p=2): return torch.mean(x.clamp_min(0) ** p) ** (1 / p)

def satagg(vals, p=2):
    v = torch.stack([x.reshape(()) for x in vals])
    return 1 - torch.mean((1 - v).clamp_min(0) ** p) ** (1 / p)


def supervised_loss(m: Model, s: Scene):
    u, b = m.unary(s.x), m.binary(s)
    losses = []
    for i, c in enumerate(COLORS): losses.append(F.binary_cross_entropy(u[f"is_{c}"], (s.color == i).float()))
    for i, f in enumerate(SHAPES): losses.append(F.binary_cross_entropy(u[f"is_{f}"], (s.shape == i).float()))
    losses += [F.binary_cross_entropy(u["is_small"], (s.size == 0).float()), F.binary_cross_entropy(u["is_big"], (s.size == 1).float())]
    targets = {"left": gt_left(s), "right": gt_right(s), "below": gt_below(s), "above": gt_above(s), "close": gt_close_soft(s), "same_size": gt_same_size(s), "can_stack": gt_can_stack(s)}
    losses += [F.binary_cross_entropy(b[k], v) for k, v in targets.items()]
    losses.append(F.binary_cross_entropy(m.ternary(s), gt_between(s)))
    return torch.stack(losses).mean()


def formulas(m: Model, s: Scene):
    n, u, b, between = s.n, m.unary(s.x), m.binary(s), m.ternary(s)
    left, right, below, above = b["left"], b["right"], b["below"], b["above"]
    close, same_size, can_stack = b["close"], b["same_size"], b["can_stack"]
    shapes = [u[f"is_{f}"] for f in SHAPES]
    unique = [forall(l_not(l_and(shapes[i], shapes[j]))) for i in range(5) for j in range(i + 1, 5)]
    between_logic = torch.zeros_like(between)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                between_logic[x, y, z] = l_or(l_and(left[y, x], right[z, x]), l_and(left[z, x], right[y, x]))
    eye = torch.eye(n, dtype=torch.bool)
    not_eye = ~eye
    return {
        "shape_unique": satagg(unique),
        "shape_coverage": forall(l_or(*shapes)),
        "left_irreflexive": forall(l_not(torch.diag(left))),
        "left_asymmetric": forall(l_imp(left, l_not(left.T))),
        "left_right_inverse": forall(l_eq(left, right.T)),
        "left_transitive": forall(l_imp(l_and(left[:, :, None], left[None, :, :]), left[:, None, :])),
        "below_above_inverse": forall(l_eq(below, above.T)),
        "below_transitive": forall(l_imp(l_and(below[:, :, None], below[None, :, :]), below[:, None, :])),
        "between_rule": forall(l_eq(between, between_logic)),
        "last_left": exists(torch.stack([forall(row) for row in left[not_eye].reshape(n, n - 1)])),
        "last_right": exists(torch.stack([forall(row) for row in right[not_eye].reshape(n, n - 1)])),
        "can_stack_rule": forall(l_imp(can_stack, l_not(l_or(u["is_cone"][None, :], u["is_triangle"][None, :])))),
        "query_small_below_cylinder_left_square": exists(l_and(u["is_small"][:, None, None], u["is_cylinder"][None, :, None], below[:, :, None], u["is_square"][None, None, :], left[:, None, :])),
        "query_green_cone_between": exists(l_and(u["is_green"][:, None, None], u["is_cone"][:, None, None], between)),
        "query_triangles_close_same_size": forall(l_imp(l_and(u["is_triangle"][:, None], u["is_triangle"][None, :], close), same_size)),
    }


def ltn_api_sat_check(m: Model, s: Scene):
    """Auditoria curta usando objetos reais do LTNtorch, quando instalado."""
    if not HAS_LTN:
        return {}
    try:
        x = ltn.Variable("x", s.x)
        y = ltn.Variable("y", s.x)
        is_circle = ltn.Predicate(m.shape["circle"])
        is_square = ltn.Predicate(m.shape["square"])
        left = ltn.Predicate(m.rel["left"])
        right = ltn.Predicate(m.rel["right"])
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg()
        axioms = [
            Forall(x, Not(And(is_circle(x), is_square(x)))),
            Forall([x, y], Implies(left(x, y), Not(left(y, x)))),
            Forall([x, y], And(Implies(left(x, y), right(y, x)), Implies(right(y, x), left(x, y)))),
            Forall(x, Or(is_circle(x), is_square(x))),
        ]
        sat = SatAgg(*axioms)
        return {"ltn_api_sat_check": float(sat.item() if hasattr(sat, "item") else sat.value.item())}
    except Exception as exc:
        print(f"Aviso: auditoria LTNtorch ignorada: {exc}")
        return {}


def train(s: Scene, epochs: int, lr: float, axiom_weight: float):
    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(epochs):
        opt.zero_grad()
        facts = supervised_loss(m, s)
        sat = satagg(formulas(m, s).values())
        loss = facts + axiom_weight * (1 - sat)
        loss.backward(); opt.step()
        if (e + 1) % 100 == 0: print(f"epoch={e+1:04d} loss={loss.item():.4f} satAgg={sat.item():.4f}")
    return m


def metrics(y_true, y_score):
    yt, yp = y_true.reshape(-1).bool(), (y_score.reshape(-1) >= 0.5)
    tp, tn = int((yt & yp).sum()), int((~yt & ~yp).sum())
    fp, fn = int((~yt & yp).sum()), int((yt & ~yp).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, EPS)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate(m: Model, s: Scene):
    with torch.no_grad():
        f, b, bt = formulas(m, s), m.binary(s), m.ternary(s)
        targets = {"left": gt_left(s), "right": gt_right(s), "below": gt_below(s), "above": gt_above(s), "close": gt_close(s), "same_size": gt_same_size(s), "can_stack": gt_can_stack(s), "between": gt_between(s)}
        scores = {**b, "between": bt}
        mt = metrics(torch.cat([targets[k].reshape(-1) for k in targets]), torch.cat([scores[k].reshape(-1) for k in targets]))
        return {"satAgg": float(satagg(f.values()).item()), **{k: float(v.item()) for k, v in f.items()}, **mt, **ltn_api_sat_check(m, s)}


def plot_scene(s: Scene, out_dir: Path):
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    markers = {"circle": "o", "square": "s", "cylinder": "D", "cone": "^", "triangle": "v"}
    cmap = {"red": "tab:red", "green": "tab:green", "blue": "tab:blue"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(s.n):
        ax.scatter(float(s.x[i, 0]), float(s.x[i, 1]), s=80 + 70 * int(s.size[i]), c=cmap[COLORS[int(s.color[i])]], marker=markers[SHAPES[int(s.shape[i])]], edgecolor="black")
        ax.text(float(s.x[i, 0]) + 0.01, float(s.x[i, 1]) + 0.01, str(i), fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title=f"CLEVR simplificado - seed {s.seed}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"scene_seed_{s.seed}.png", dpi=160)
    plt.close(fig)


def run(args):
    rows = []
    for i in range(args.runs):
        seed = args.seed + i
        print(f"\n=== Execucao {i+1}/{args.runs} | seed={seed} ===")
        seed_all(seed)
        s = generate_scene(seed, args.n_objects)
        if args.plot_dir:
            plot_scene(s, Path(args.plot_dir))
        m = train(s, args.epochs, args.lr, args.axiom_weight)
        row = {"run": i + 1, "seed": seed, **evaluate(m, s)}
        rows.append(row)
        print("satAgg={satAgg:.4f} acc={accuracy:.4f} prec={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(**row))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=350)
    p.add_argument("--n-objects", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--axiom-weight", type=float, default=0.35)
    p.add_argument("--out", default="resultados_clevr_ltn.csv")
    p.add_argument("--plot-dir", default="")
    args = p.parse_args()
    rows = run(args)
    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for k in ["satAgg", "accuracy", "precision", "recall", "f1"]:
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"{k}: media={v.mean():.4f} desvio={v.std():.4f}")


if __name__ == "__main__":
    main()
