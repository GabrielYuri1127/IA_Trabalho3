"""Trabalho 3 FIA - CLEVR simplificado com raciocinio neuro-simbolico.

Rode com:
    pip install torch numpy matplotlib LTNtorch
    python clevr_ltn_experimentos.py --runs 5 --epochs 350 --min-distance 0.08 --overlap-margin 0.012 --out resultados_clevr_ltn.csv --plot-dir figuras

O codigo treina em uma cena CLEVR simplificada balanceada e testa em 5 cenas
aleatorias distintas. Cada cena padrao tem 25 objetos: 5 classes de formas
(circulo, quadrado, elipse, retangulo e triangulo), com 5 objetos de cada classe.
Em seguida calcula satisfatibilidade das formulas pedidas no enunciado e
reporta acuracia, precisao, recall e F1.
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
SHAPES = ["circle", "square", "ellipse", "rectangle", "triangle"]
EPS = 1e-7
DEFAULT_MIN_DISTANCE = 0.08
DEFAULT_OVERLAP_MARGIN = 0.012
DEFAULT_CLOSE_THRESHOLD = 0.25


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


def balanced_shape_indices(rng: np.random.Generator, n: int) -> np.ndarray:
    base = n // len(SHAPES)
    remainder = n % len(SHAPES)
    shape = np.repeat(np.arange(len(SHAPES)), base)
    if remainder:
        shape = np.concatenate([shape, rng.choice(len(SHAPES), size=remainder, replace=False)])
    rng.shuffle(shape)
    return shape.astype(np.int64)


def visual_half_extents(shape_idx: int, size_idx: int) -> tuple[float, float]:
    """Half-width/half-height used by the generated 2D plot for overlap checks."""
    scale = 0.055 + 0.025 * int(size_idx)
    shape = SHAPES[int(shape_idx)]
    if shape == "circle":
        r = scale * 0.45
        return r, r
    if shape == "square":
        h = scale * 0.85 / 2
        return h, h
    if shape in {"ellipse", "rectangle"}:
        return scale * 1.35 / 2, scale * 0.75 / 2
    r = scale * 0.55
    return r, r


def bbox_gap(pos_a: torch.Tensor, ext_a: tuple[float, float], pos_b: torch.Tensor, ext_b: tuple[float, float]) -> float:
    gap_x = abs(float(pos_a[0] - pos_b[0])) - (ext_a[0] + ext_b[0])
    gap_y = abs(float(pos_a[1] - pos_b[1])) - (ext_a[1] + ext_b[1])
    return max(gap_x, gap_y)


def non_overlapping_positions(
    rng: np.random.Generator,
    shape: np.ndarray,
    size: np.ndarray,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    overlap_margin: float = DEFAULT_OVERLAP_MARGIN,
) -> np.ndarray:
    n = len(shape)
    extents = [visual_half_extents(int(shape[i]), int(size[i])) for i in range(n)]
    positions: list[np.ndarray] = []
    for i in range(n):
        hx, hy = extents[i]
        for _attempt in range(5000):
            candidate = np.array([
                rng.uniform(hx + overlap_margin, 1 - hx - overlap_margin),
                rng.uniform(hy + overlap_margin, 1 - hy - overlap_margin),
            ], dtype=np.float32)
            if not positions:
                positions.append(candidate)
                break
            dist = np.linalg.norm(np.stack(positions) - candidate, axis=1)
            center_ok = float(dist.min()) >= min_distance
            bbox_ok = all(
                bbox_gap(torch.tensor(candidate), extents[i], torch.tensor(prev), extents[j]) >= overlap_margin
                for j, prev in enumerate(positions)
            )
            if center_ok and bbox_ok:
                positions.append(candidate)
                break
        else:
            raise RuntimeError(
                f"Nao foi possivel gerar cena sem overlapping com min_distance={min_distance} "
                f"e overlap_margin={overlap_margin}"
            )
    return np.stack(positions).astype(np.float32)


def generate_scene(
    seed: int,
    n: int = 25,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    overlap_margin: float = DEFAULT_OVERLAP_MARGIN,
) -> Scene:
    rng = np.random.default_rng(seed)
    color = rng.integers(0, 3, size=n)
    shape = balanced_shape_indices(rng, n)
    size = rng.integers(0, 2, size=n)
    pos = non_overlapping_positions(rng, shape, size, min_distance, overlap_margin)
    x = np.concatenate([pos, one_hot(color, 3), one_hot(shape, 5), size[:, None]], axis=1)
    return Scene(torch.tensor(x).float(), torch.tensor(color), torch.tensor(shape), torch.tensor(size), seed)


def shape_counts(s: Scene) -> dict[str, int]:
    return {f"n_{shape}": int((s.shape == i).sum().item()) for i, shape in enumerate(SHAPES)}


def min_pair_distance(s: Scene) -> float:
    dist = torch.cdist(s.x[:, :2], s.x[:, :2])
    dist.fill_diagonal_(float("inf"))
    return float(dist.min().item())


def min_bbox_gap(s: Scene) -> float:
    gaps = []
    extents = [visual_half_extents(int(s.shape[i]), int(s.size[i])) for i in range(s.n)]
    for i in range(s.n):
        for j in range(i + 1, s.n):
            gaps.append(bbox_gap(s.x[i, :2], extents[i], s.x[j, :2], extents[j]))
    return float(min(gaps)) if gaps else float("inf")


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


def gt_close(s: Scene, threshold: float = DEFAULT_CLOSE_THRESHOLD):
    dist = torch.cdist(s.x[:, :2], s.x[:, :2])
    return ((dist < threshold) & (dist > 0)).float()


def gt_same_size(s: Scene):
    return (s.size[:, None] == s.size[None, :]).float()


def gt_can_stack(s: Scene):
    support_shape = s.shape[None, :]
    support_ok = ~((support_shape == SHAPES.index("rectangle")) | (support_shape == SHAPES.index("triangle")))
    above = gt_above(s).bool()
    same_size = s.size[:, None] == s.size[None, :]
    aligned = torch.abs(s.x[:, None, 0] - s.x[None, :, 0]) < 0.15
    return (above & support_ok & (same_size | aligned)).float()


def gt_horiz_aligned(s: Scene):
    delta = torch.abs(s.x[:, None, 0] - s.x[None, :, 0])
    return torch.exp(-((delta / 0.15) ** 2)).clamp(EPS, 1 - EPS)


def gt_between(s: Scene):
    left = gt_left(s).bool()
    right = gt_right(s).bool()
    n = s.n
    idx = torch.arange(n)
    x = idx[:, None, None]
    y = idx[None, :, None]
    z = idx[None, None, :]
    distinct = (x != y) & (x != z) & (y != z)
    cond1 = left.T[:, :, None] & right.T[:, None, :]
    cond2 = left.T[:, None, :] & right.T[:, :, None]
    return ((cond1 | cond2) & distinct).float()


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
    targets = {"left": gt_left(s), "right": gt_right(s), "below": gt_below(s), "above": gt_above(s), "close": gt_close(s), "same_size": gt_same_size(s), "can_stack": gt_can_stack(s)}
    losses += [F.binary_cross_entropy(b[k], v) for k, v in targets.items()]
    losses.append(F.binary_cross_entropy(m.ternary(s), gt_between(s)))
    return torch.stack(losses).mean()


def formulas(m: Model, s: Scene):
    n, u, b, between = s.n, m.unary(s.x), m.binary(s), m.ternary(s)
    left, right, below, above = b["left"], b["right"], b["below"], b["above"]
    close, same_size, can_stack = b["close"], b["same_size"], b["can_stack"]
    shapes = [u[f"is_{f}"] for f in SHAPES]
    unique = [forall(l_not(l_and(shapes[i], shapes[j]))) for i in range(5) for j in range(i + 1, 5)]
    between_logic = l_or(l_and(left.T[:, :, None], right.T[:, None, :]), l_and(left.T[:, None, :], right.T[:, :, None]))
    horiz_aligned = gt_horiz_aligned(s)
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
        "can_stack_above_rule": forall(l_imp(can_stack, above)),
        "can_stack_rule": forall(l_imp(can_stack, l_not(l_or(u["is_rectangle"][None, :], u["is_triangle"][None, :])))),
        "can_stack_stability_rule": forall(l_imp(can_stack, l_or(same_size, horiz_aligned))),
        "query_small_below_ellipse_left_square": exists(l_and(u["is_small"][:, None, None], u["is_ellipse"][None, :, None], below[:, :, None], u["is_square"][None, None, :], left[:, None, :])),
        "query_green_rectangle_between": exists(l_and(u["is_green"][:, None, None], u["is_rectangle"][:, None, None], between)),
        "query_triangles_close_same_size": forall(l_imp(l_and(u["is_triangle"][:, None], u["is_triangle"][None, :], close), same_size)),
    }


def query_evidence(s: Scene, u: dict[str, torch.Tensor], b: dict[str, torch.Tensor], between: torch.Tensor):
    n = s.n

    below_gt = gt_below(s).bool()
    left_gt = gt_left(s).bool()
    between_gt = gt_between(s).bool()
    close_gt = gt_close(s).bool()
    same_size_gt = gt_same_size(s).bool()

    is_small_gt = s.size == 0
    is_green_gt = s.color == COLORS.index("green")
    is_square_gt = s.shape == SHAPES.index("square")
    is_ellipse_gt = s.shape == SHAPES.index("ellipse")
    is_rectangle_gt = s.shape == SHAPES.index("rectangle")
    is_triangle_gt = s.shape == SHAPES.index("triangle")

    q1_gt = (
        is_small_gt[:, None, None]
        & is_ellipse_gt[None, :, None]
        & below_gt[:, :, None]
        & is_square_gt[None, None, :]
        & left_gt[:, None, :]
    )
    q1_score = l_and(
        u["is_small"][:, None, None],
        u["is_ellipse"][None, :, None],
        b["below"][:, :, None],
        u["is_square"][None, None, :],
        b["left"][:, None, :],
    )
    q1_flat = int(torch.argmax(q1_score).item())
    q1_x = q1_flat // (n * n)

    q2_gt = is_rectangle_gt[:, None, None] & is_green_gt[:, None, None] & between_gt
    q2_score = l_and(u["is_rectangle"][:, None, None], u["is_green"][:, None, None], between)
    q2_flat = int(torch.argmax(q2_score).item())
    q2_x = q2_flat // (n * n)

    q3_pairs = is_triangle_gt[:, None] & is_triangle_gt[None, :] & close_gt
    q3_violations = q3_pairs & ~same_size_gt

    return {
        "q1_gt_exists": int(q1_gt.any().item()),
        "q1_best_witness_conf": float(q1_score.max().item()),
        "q1_best_witness_obj": int(q1_x),
        "q2_gt_exists": int(q2_gt.any().item()),
        "q2_best_witness_conf": float(q2_score.max().item()),
        "q2_best_witness_obj": int(q2_x),
        "q3_triangle_close_pairs": int(q3_pairs.sum().item()),
        "q3_triangle_size_violations": int(q3_violations.sum().item()),
    }


def xai_pair_evidence(s: Scene, b: dict[str, torch.Tensor]):
    leftmost = int(torch.argmin(s.x[:, 0]).item())
    rightmost = int(torch.argmax(s.x[:, 0]).item())
    forward = b["left"][leftmost, rightmost]
    reverse = b["left"][rightmost, leftmost]
    asymmetry = l_imp(forward, l_not(reverse))
    return {
        "xai_leftmost_obj": leftmost,
        "xai_rightmost_obj": rightmost,
        "xai_leftmost_rightmost_conf": float(forward.item()),
        "xai_reverse_conf": float(reverse.item()),
        "xai_asymmetry_conf": float(asymmetry.item()),
    }


def ltn_api_sat_check(m: Model, s: Scene):
    """Auditoria curta usando objetos reais do LTNtorch, quando instalado."""
    if not HAS_LTN:
        return {}
    try:
        x = ltn.Variable("x", s.x)
        y = ltn.Variable("y", s.x)
        shape_preds = [ltn.Predicate(m.shape[name]) for name in SHAPES]
        left = ltn.Predicate(m.rel["left"])
        right = ltn.Predicate(m.rel["right"])
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg()

        def or_many(values):
            out = values[0]
            for value in values[1:]:
                out = Or(out, value)
            return out

        shape_axioms = []
        for i in range(len(shape_preds)):
            for j in range(i + 1, len(shape_preds)):
                shape_axioms.append(Forall(x, Not(And(shape_preds[i](x), shape_preds[j](x)))))
        axioms = [
            *shape_axioms,
            Forall(x, or_many([pred(x) for pred in shape_preds])),
            Forall([x, y], Implies(left(x, y), Not(left(y, x)))),
            Forall([x, y], And(Implies(left(x, y), right(y, x)), Implies(right(y, x), left(x, y)))),
        ]
        sat = SatAgg(*axioms)
        return {"ltn_api_sat_check": float(sat.item() if hasattr(sat, "item") else sat.value.item())}
    except Exception as exc:
        print(f"Aviso: auditoria LTNtorch ignorada: {exc}")
        return {}


def ltn_value(obj):
    return obj if torch.is_tensor(obj) else obj.value


def ltn_training_sat(m: Model, s: Scene):
    """Termo diferenciavel de satisfatibilidade usando objetos do LTNtorch."""
    if not HAS_LTN:
        raise RuntimeError("LTNtorch e obrigatorio para o termo SatAgg usado no treinamento.")

    x = ltn.Variable("x_train", s.x)
    y = ltn.Variable("y_train", s.x)
    shape_preds = [ltn.Predicate(m.shape[name]) for name in SHAPES]
    left = ltn.Predicate(m.rel["left"])
    right = ltn.Predicate(m.rel["right"])

    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    def or_many(values):
        out = values[0]
        for value in values[1:]:
            out = Or(out, value)
        return out

    shape_axioms = [
        Forall(x, Not(And(shape_preds[i](x), shape_preds[j](x))))
        for i in range(len(shape_preds))
        for j in range(i + 1, len(shape_preds))
    ]
    axioms = [
        *shape_axioms,
        Forall(x, or_many([pred(x) for pred in shape_preds])),
        Forall([x, y], Implies(left(x, y), Not(left(y, x)))),
        Forall([x, y], And(Implies(left(x, y), right(y, x)), Implies(right(y, x), left(x, y)))),
    ]
    return ltn_value(SatAgg(*axioms)).clamp(EPS, 1)


def train(s: Scene, epochs: int, lr: float, axiom_weight: float):
    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ltn_training_active = False
    last_ltn_sat = None
    for e in range(epochs):
        opt.zero_grad()
        facts = supervised_loss(m, s)
        custom_sat = satagg(formulas(m, s).values())
        ltn_sat = ltn_training_sat(m, s)
        ltn_training_active = True
        last_ltn_sat = float(ltn_sat.detach().item())
        sat = satagg([custom_sat, ltn_sat])
        loss = facts + axiom_weight * (1 - sat)
        loss.backward(); opt.step()
        if (e + 1) % 100 == 0: print(f"epoch={e+1:04d} loss={loss.item():.4f} satAgg={sat.item():.4f}")
    m.ltn_training_active = int(ltn_training_active)
    m.ltn_training_sat_final = last_ltn_sat if last_ltn_sat is not None else float("nan")
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


def prefixed_metrics(prefix: str, y_true: torch.Tensor, y_score: torch.Tensor) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics(y_true, y_score).items()}


def unary_targets_and_scores(m: Model, s: Scene):
    u = m.unary(s.x)
    targets, scores = [], []
    for i, c in enumerate(COLORS):
        targets.append((s.color == i).float().reshape(-1))
        scores.append(u[f"is_{c}"].reshape(-1))
    for i, f in enumerate(SHAPES):
        targets.append((s.shape == i).float().reshape(-1))
        scores.append(u[f"is_{f}"].reshape(-1))
    targets += [(s.size == 0).float().reshape(-1), (s.size == 1).float().reshape(-1)]
    scores += [u["is_small"].reshape(-1), u["is_big"].reshape(-1)]
    return torch.cat(targets), torch.cat(scores)


def evaluate(m: Model, s: Scene):
    with torch.no_grad():
        f, u, b, bt = formulas(m, s), m.unary(s.x), m.binary(s), m.ternary(s)
        targets = {"left": gt_left(s), "right": gt_right(s), "below": gt_below(s), "above": gt_above(s), "close": gt_close(s), "same_size": gt_same_size(s), "can_stack": gt_can_stack(s), "between": gt_between(s)}
        scores = {**b, "between": bt}
        relation_true = torch.cat([targets[k].reshape(-1) for k in targets])
        relation_score = torch.cat([scores[k].reshape(-1) for k in targets])
        unary_true, unary_score = unary_targets_and_scores(m, s)
        relation_mt = metrics(relation_true, relation_score)
        all_mt = prefixed_metrics("all", torch.cat([unary_true, relation_true]), torch.cat([unary_score, relation_score]))
        unary_mt = prefixed_metrics("unary", unary_true, unary_score)
        relation_prefixed = prefixed_metrics("relation", relation_true, relation_score)
        return {
            "satAgg": float(satagg(f.values()).item()),
            **{k: float(v.item()) for k, v in f.items()},
            **query_evidence(s, u, b, bt),
            **xai_pair_evidence(s, b),
            **relation_mt,
            **all_mt,
            **unary_mt,
            **relation_prefixed,
            **ltn_api_sat_check(m, s),
        }


def plot_scene(s: Scene, out_dir: Path):
    if plt is None:
        return
    from matplotlib.patches import Circle, Ellipse, Rectangle, RegularPolygon

    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = {"red": "tab:red", "green": "tab:green", "blue": "tab:blue"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(s.n):
        x = float(s.x[i, 0])
        y = float(s.x[i, 1])
        shape = SHAPES[int(s.shape[i])]
        color = cmap[COLORS[int(s.color[i])]]
        scale = 0.055 + 0.025 * int(s.size[i])

        if shape == "circle":
            patch = Circle((x, y), radius=scale * 0.45, facecolor=color, edgecolor="black")
        elif shape == "square":
            side = scale * 0.85
            patch = Rectangle((x - side / 2, y - side / 2), side, side, facecolor=color, edgecolor="black")
        elif shape == "ellipse":
            patch = Ellipse((x, y), width=scale * 1.35, height=scale * 0.75, facecolor=color, edgecolor="black")
        elif shape == "rectangle":
            width, height = scale * 1.35, scale * 0.75
            patch = Rectangle((x - width / 2, y - height / 2), width, height, facecolor=color, edgecolor="black")
        else:
            patch = RegularPolygon((x, y), numVertices=3, radius=scale * 0.55, orientation=np.pi / 2, facecolor=color, edgecolor="black")

        ax.add_patch(patch)
        ax.text(x + 0.012, y + 0.012, str(i), fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title=f"CLEVR simplificado - seed {s.seed}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"scene_seed_{s.seed}.png", dpi=160)
    plt.close(fig)


def run(args):
    rows = []
    print(f"\n=== Treinamento unico | train_seed={args.train_seed} ===")
    seed_all(args.train_seed)
    train_scene = generate_scene(args.train_seed, args.n_objects, args.min_distance, args.overlap_margin)
    if args.plot_dir:
        plot_scene(train_scene, Path(args.plot_dir))
    model = train(train_scene, args.epochs, args.lr, args.axiom_weight)

    for i in range(args.runs):
        seed = args.seed + i
        print(f"\n=== Teste {i+1}/{args.runs} | train_seed={args.train_seed} | test_seed={seed} ===")
        seed_all(seed)
        s = generate_scene(seed, args.n_objects, args.min_distance, args.overlap_margin)
        if args.plot_dir:
            plot_scene(s, Path(args.plot_dir))
        row = {
            "run": i + 1,
            "train_seed": args.train_seed,
            "seed": seed,
            **shape_counts(s),
            "min_pair_distance": min_pair_distance(s),
            "min_bbox_gap": min_bbox_gap(s),
            "overlap_ok": int(min_pair_distance(s) >= args.min_distance),
            "bbox_overlap_ok": int(min_bbox_gap(s) >= args.overlap_margin),
            "close_threshold": DEFAULT_CLOSE_THRESHOLD,
            "close_training_aligned": 1,
            "ltn_training_active": getattr(model, "ltn_training_active", 0),
            "ltn_training_sat_final": getattr(model, "ltn_training_sat_final", float("nan")),
            **evaluate(model, s),
        }
        rows.append(row)
        print("satAgg={satAgg:.4f} acc={accuracy:.4f} prec={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(**row))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=350)
    p.add_argument("--n-objects", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-seed", type=int, default=2025)
    p.add_argument("--min-distance", type=float, default=DEFAULT_MIN_DISTANCE)
    p.add_argument("--overlap-margin", type=float, default=DEFAULT_OVERLAP_MARGIN)
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
