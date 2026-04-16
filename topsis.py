from dataclasses import dataclass
from typing import List, Optional, Sequence, Literal
import numpy as np
import pandas as pd

ImpactType = Literal["benefit", "cost"]

@dataclass
class TopsisResult:
    X: pd.DataFrame
    R: pd.DataFrame
    V: pd.DataFrame
    ideal_best: pd.Series
    ideal_worst: pd.Series
    D_plus: pd.Series
    D_minus: pd.Series
    C: pd.Series
    ranking: pd.DataFrame

def _normalize_weights(weights: Sequence[float], n: int) -> np.ndarray:
    w = np.array(weights, dtype=float)
    if len(w) != n:
        raise ValueError("權重數量必須等於指標數量")
    if np.any(w < 0):
        raise ValueError("權重不可為負數")
    s = w.sum()
    if s == 0:
        raise ValueError("權重總和不可為0")
    return w / s

def topsis(
    df: pd.DataFrame,
    alt_col: str,
    criteria_cols: List[str],
    impacts: Sequence[ImpactType],
    weights: Optional[Sequence[float]] = None,
):
    data = df[[alt_col] + criteria_cols].copy()

    for c in criteria_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna()

    alts = data[alt_col]
    X = data[criteria_cols].astype(float)
    X.index = alts

    # Step 1: 正規化
    denom = np.sqrt((X ** 2).sum())
    R = X / denom

    # Step 2: 權重
    n = len(criteria_cols)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = _normalize_weights(weights, n)

    V = R * w

    # Step 3: 理想解
    ideal_best = []
    ideal_worst = []

    for i, imp in enumerate(impacts):
        col = V.iloc[:, i]
        if imp == "benefit":
            ideal_best.append(col.max())
            ideal_worst.append(col.min())
        else:
            ideal_best.append(col.min())
            ideal_worst.append(col.max())

    ideal_best = pd.Series(ideal_best, index=criteria_cols)
    ideal_worst = pd.Series(ideal_worst, index=criteria_cols)

    # Step 4: 距離
    D_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # Step 5: 接近度
    C = D_minus / (D_plus + D_minus)

    ranking = pd.DataFrame({
        alt_col: alts,
        "D+": D_plus,
        "D-": D_minus,
        "C": C
    }).sort_values("C", ascending=False)

    ranking.insert(0, "Rank", range(1, len(ranking) + 1))

    return TopsisResult(X, R, V, ideal_best, ideal_worst, D_plus, D_minus, C, ranking)
