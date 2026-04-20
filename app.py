import pandas as pd
import streamlit as st
import plotly.express as px

from topsis import run_topsis

st.set_page_config(page_title="TOPSIS 分析網站", layout="centered")

st.title("TOPSIS 分析網站")

# =====================
# 讀檔
# =====================
def load_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="big5")

    elif name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    elif name.endswith(".ods"):
        return pd.read_excel(uploaded_file, engine="odf")

# =====================
# 上傳 / sample
# =====================
uploaded_file = st.file_uploader("上傳資料（CSV / Excel / ODS）")

st.caption("未上傳時會自動使用 sample.csv")

if uploaded_file is not None:
    df = load_file(uploaded_file)
    st.success("已載入上傳資料")
else:
    df = pd.read_csv("sample.csv", encoding="utf-8-sig")
    st.info("使用示範資料 sample.csv")

df.columns = df.columns.str.strip()
df = df.dropna(how="all")

st.subheader("資料預覽")
st.dataframe(df, width=900, height=200)

# =====================
# 欄位選擇
# =====================
cols = df.columns.tolist()

id_col = st.selectbox("選擇識別欄位", cols)

metric_cols = st.multiselect(
    "選擇指標（至少2項）",
    [c for c in cols if c != id_col]
)

if len(metric_cols) < 2:
    st.stop()

# =====================
# 數值轉換
# =====================
for col in metric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=metric_cols).reset_index(drop=True)

# =====================
# 成本 / 效益
# =====================
st.subheader("指標類型")

metric_types = {}
for col in metric_cols:
    metric_types[col] = st.radio(
        col,
        ["成本型", "效益型"],
        horizontal=True
    )

# =====================
# 權重
# =====================
st.subheader("權重設定")

weights = {}
for col in metric_cols:
    weights[col] = st.number_input(col, value=1.0)

if st.button("開始分析"):

    benefit_flags = ["效益" in metric_types[c] for c in metric_cols]
    weight_list = [weights[c] for c in metric_cols]

    result = run_topsis(
        df=df,
        id_col=id_col,
        criteria_cols=metric_cols,
        weights=weight_list,
        benefit_flags=benefit_flags
    )

    # =====================
    # 🏆 排名（最重要）
    # =====================
    st.subheader("🏆 最終排名")

    st.dataframe(result["result_df"], width=900, height=260)

    # TOP3摘要
    top3 = result["result_df"].head(min(3, len(result["result_df"])))

    if len(top3) >= 3:
        st.success(
            f"🥇 {top3.iloc[0][id_col]}（C={top3.iloc[0]['C']:.4f}）\n"
            f"🥈 {top3.iloc[1][id_col]}（C={top3.iloc[1]['C']:.4f}）\n"
            f"🥉 {top3.iloc[2][id_col]}（C={top3.iloc[2]['C']:.4f}）"
        )

    # =====================
    # 圖表
    # =====================
    st.subheader("📊 C值排名圖")

    fig = px.bar(
        result["result_df"],
        x=id_col,
        y="C",
        text="C"
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")

    st.plotly_chart(fig)

    # =====================
    # 詳細資料（收起）
    # =====================
    with st.expander("查看計算細節"):

        st.write("R矩陣")
        st.dataframe(result["R_df"])

        st.write("V矩陣")
        st.dataframe(result["V_df"])

        st.write("理想解 / 負理想解")
        ideal_df = pd.DataFrame({
            "指標": metric_cols,
            "A+": result["A_plus"],
            "A-": result["A_minus"]
        })
        st.dataframe(ideal_df)
