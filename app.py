import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from topsis import run_topsis

st.set_page_config(page_title="TOPSIS 分析網站", layout="wide")

st.title("TOPSIS 分析網站")
st.write("可上傳 CSV / Excel / ODS，選擇識別欄位、分析指標、成本/效益型與權重後，自動完成 TOPSIS 分析。")

# ---------------------------
# 讀檔函式
# ---------------------------
def load_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        # 先嘗試 utf-8，不行再用 big5
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="big5")
        return df

    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    elif name.endswith(".ods"):
        return pd.read_excel(uploaded_file, engine="odf")

    else:
        raise ValueError("不支援的檔案格式，請上傳 csv / xlsx / ods")

# ---------------------------
# 工具函式
# ---------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    return df

def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# ---------------------------
# 上傳資料
# ---------------------------
uploaded_file = st.file_uploader(
    "請上傳資料檔案（CSV / Excel / ODS）",
    type=["csv", "xlsx", "xls", "ods"]
)

if uploaded_file is None:
    st.info("請先上傳資料檔案。")
    st.stop()

try:
    df_raw = load_file(uploaded_file)
    df_raw = clean_dataframe(df_raw)
except Exception as e:
    st.error(f"讀取檔案失敗：{e}")
    st.stop()

st.subheader("原始資料預覽")
st.dataframe(df_raw, use_container_width=True)

all_columns = df_raw.columns.tolist()

# ---------------------------
# 基本設定
# ---------------------------
st.subheader("欄位設定")

id_col = st.selectbox(
    "請選擇『識別欄位』（例如：城市、縣市、學校、公司、方案名稱）",
    all_columns
)

# 可選年份欄位（非必要）
use_group = st.checkbox("是否依某欄位分組分析（例如：年份）", value=False)

group_col = None
group_values = None
if use_group:
    group_col = st.selectbox("請選擇分組欄位", [c for c in all_columns if c != id_col])
    group_values = df_raw[group_col].dropna().unique().tolist()
    group_values = sorted(group_values, key=lambda x: str(x))

# ---------------------------
# 指標選擇
# ---------------------------
st.subheader("指標設定")

candidate_metric_cols = [c for c in all_columns if c not in [id_col, group_col]]
metric_cols = st.multiselect(
    "請選擇要分析的指標欄位（至少 2 項）",
    candidate_metric_cols
)

if len(metric_cols) < 2:
    st.warning("請至少選擇 2 個指標。")
    st.stop()

# 將所選欄位轉數值，並提示缺值狀況
df_work = df_raw.copy()
for col in metric_cols:
    df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

# 顯示缺值統計
with st.expander("查看缺值統計"):
    na_info = df_work[[id_col] + metric_cols].isna().sum().reset_index()
    na_info.columns = ["欄位", "缺值數量"]
    st.dataframe(na_info, use_container_width=True)

st.subheader("成本型 / 效益型設定")
metric_types = {}
for col in metric_cols:
    metric_types[col] = st.radio(
        f"{col} 屬於哪一種？",
        options=["成本型（越小越好）", "效益型（越大越好）"],
        horizontal=True,
        key=f"type_{col}"
    )

st.subheader("權重設定")
st.caption("可以手動調整權重，系統會自動標準化，使總和 = 1。")

weights = {}
weight_cols = st.columns(min(4, len(metric_cols)))
for i, col in enumerate(metric_cols):
    with weight_cols[i % len(weight_cols)]:
        weights[col] = st.number_input(
            f"{col} 權重",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"weight_{col}"
        )

if sum(weights.values()) == 0:
    st.error("權重總和不能為 0。")
    st.stop()

# ---------------------------
# 執行按鈕
# ---------------------------
run_btn = st.button("開始分析", type="primary")

if not run_btn:
    st.stop()

# ---------------------------
# 分析函式
# ---------------------------
def analyze_one_dataframe(df_input: pd.DataFrame, title: str):
    st.markdown("---")
    st.header(title)

    cols_needed = [id_col] + metric_cols
    df_sub = df_input[cols_needed].copy()

    # 清理識別欄位
    df_sub[id_col] = df_sub[id_col].astype(str).str.strip()
    df_sub = df_sub[df_sub[id_col] != ""]
    df_sub = df_sub[df_sub[id_col].str.lower() != "none"]
    df_sub = df_sub[df_sub[id_col].str.lower() != "nan"]

    # 丟掉有缺值的列
    df_sub = df_sub.dropna(subset=metric_cols).reset_index(drop=True)

    if df_sub.empty:
        st.warning("此分組沒有可用資料。")
        return

    # 如果識別欄位有重複，做平均
    if df_sub[id_col].duplicated().any():
        st.info(f"偵測到「{id_col}」有重複值，系統將對同名項目自動取平均。")
        df_sub = df_sub.groupby(id_col, as_index=False)[metric_cols].mean()

    benefit_flags = []
    for col in metric_cols:
        if "效益型" in metric_types[col]:
            benefit_flags.append(True)
        else:
            benefit_flags.append(False)

    weight_list = [weights[col] for col in metric_cols]

    result = run_topsis(
        df=df_sub,
        id_col=id_col,
        criteria_cols=metric_cols,
        weights=weight_list,
        benefit_flags=benefit_flags
    )

    st.subheader("整理後的分析資料")
    st.dataframe(result["input_df"], use_container_width=True)

    st.subheader("正規化矩陣 R")
    st.dataframe(result["R_df"], use_container_width=True)

    st.subheader("加權矩陣 V")
    st.dataframe(result["V_df"], use_container_width=True)

    st.subheader("理想解與負理想解")
    ideal_df = pd.DataFrame({
        "指標": metric_cols,
        "A+": result["A_plus"],
        "A-": result["A_minus"]
    })
    st.dataframe(ideal_df, use_container_width=True)

    st.subheader("排名結果")
    st.dataframe(result["result_df"], use_container_width=True)

    st.download_button(
        label="下載排名結果 CSV",
        data=to_csv_download(result["result_df"]),
        file_name=f"{title}_ranking.csv",
        mime="text/csv"
    )

    # 長條圖
    st.subheader("TOPSIS C值排名長條圖")
    plot_df = result["result_df"].copy()
    plot_df[id_col] = plot_df[id_col].astype(str)
    plot_df["C"] = pd.to_numeric(plot_df["C"], errors="coerce")
    plot_df = plot_df.dropna(subset=[id_col, "C"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df[id_col], plot_df["C"])
    ax.set_title("TOPSIS C值排名長條圖")
    ax.set_xlabel(id_col)
    ax.set_ylabel("C值")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------------------
# 單次分析 / 分組分析
# ---------------------------
try:
    if use_group:
        for g in group_values:
            df_group = df_work[df_work[group_col] == g].copy()
            analyze_one_dataframe(df_group, title=f"{group_col} = {g}")
    else:
        analyze_one_dataframe(df_work, title="整體分析結果")

except Exception as e:
    st.error(f"分析過程發生錯誤：{e}")
