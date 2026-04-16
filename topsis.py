import numpy as np
import pandas as pd


def run_topsis(df, id_col, criteria_cols, weights, benefit_flags):
    """
    df: 已清理好的 DataFrame
    id_col: 識別欄位名稱
    criteria_cols: 指標欄位列表
    weights: 權重列表（可未正規化）
    benefit_flags: True=效益型, False=成本型
    """

    work_df = df.copy()

    X = work_df[criteria_cols].astype(float).values

    # 權重標準化
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    # 向量正規化
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1e-12
    R = X / denom

    # 加權
    V = R * weights

    A_plus = []
    A_minus = []

    for j, is_benefit in enumerate(benefit_flags):
        col = V[:, j]
        if is_benefit:
            A_plus.append(col.max())
            A_minus.append(col.min())
        else:
            A_plus.append(col.min())
            A_minus.append(col.max())

    A_plus = np.array(A_plus)
    A_minus = np.array(A_minus)

    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    C = D_minus / (D_plus + D_minus)

    result_df = pd.DataFrame({
        id_col: work_df[id_col].values,
        "D+": D_plus,
        "D-": D_minus,
        "C": C
    })

    result_df = result_df.sort_values("C", ascending=False).reset_index(drop=True)
    result_df.insert(0, "Rank", range(1, len(result_df) + 1))

    R_df = pd.DataFrame(R, columns=[f"R_{c}" for c in criteria_cols])
    R_df.insert(0, id_col, work_df[id_col].values)

    V_df = pd.DataFrame(V, columns=[f"V_{c}" for c in criteria_cols])
    V_df.insert(0, id_col, work_df[id_col].values)

    return {
        "input_df": work_df.round(6),
        "R_df": R_df.round(6),
        "V_df": V_df.round(6),
        "A_plus": np.round(A_plus, 6),
        "A_minus": np.round(A_minus, 6),
        "result_df": result_df.round(6)
    }
