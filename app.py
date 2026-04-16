import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from topsis import topsis

st.set_page_config(page_title="TOPSIS 分析網站", layout="wide")
st.title("TOPSIS 多指標決策分析")

uploaded = st.file_uploader("上傳 CSV 或 Excel 檔", type=["csv", "xlsx"])


def read_table_auto(file):
    filename = file.name.lower()

    # Excel
    if filename.endswith(".xlsx"):
        return pd.read_excel(file)

    # CSV
    raw = file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "gbk"]

    for enc in encodings:
        try:
            text = raw.decode(enc)
            return pd.read_csv(StringIO(text))
        except Exception:
            continue

    raise ValueError("無法辨識這個檔案的格式或編碼，請確認是 .csv 或 .xlsx")


if uploaded:
    try:
        df = read_table_auto(uploaded)
    except Exception as e:
        st.error(f"讀取檔案失敗：{e}")
        st.stop()

    st.subheader("資料預覽")
    st.dataframe(df, use_container_width=True)

    columns = list(df.columns)

    alt_col = st.selectbox("選擇方案名稱欄位", columns)

    criteria_cols = st.multiselect(
        "選擇指標欄位",
        [c for c in columns if c != alt_col]
    )

    if criteria_cols:
        st.subheader("設定指標類型")

        impacts = []
        for c in criteria_cols:
            impact = st.selectbox(
                f"{c} 是成本型還是效益型？",
                ["cost", "benefit"],
                key=f"impact_{c}"
            )
            impacts.append(impact)

        if st.button("開始計算 TOPSIS"):
            try:
                result = topsis(
                    df=df,
                    alt_col=alt_col,
                    criteria_cols=criteria_cols,
                    impacts=impacts
                )

                st.success("計算完成！")

                # 排名結果
                st.subheader("排名結果")
                st.dataframe(result.ranking, use_container_width=True)

                # 下載排名 CSV
                csv_data = result.ranking.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="下載排名結果 CSV",
                    data=csv_data,
                    file_name="topsis_ranking.csv",
                    mime="text/csv"
                )

                # -------------------------
                # 圖表 1：TOPSIS C值排名長條圖
                # -------------------------
                st.subheader("TOPSIS C值排名長條圖")

                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.bar(result.ranking[alt_col], result.ranking["C"])
                ax1.set_title("TOPSIS C值排名")
                ax1.set_xlabel("方案")
                ax1.set_ylabel("C值")
                plt.xticks(rotation=45)
                st.pyplot(fig1)

                # -------------------------
                # 圖表 2：各方案原始指標比較圖
                # -------------------------
                st.subheader("各方案原始指標比較圖")

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                result.X.plot(kind="bar", ax=ax2)
                ax2.set_title("各方案原始指標比較")
                ax2.set_xlabel("方案")
                ax2.set_ylabel("數值")
                plt.xticks(rotation=45)
                st.pyplot(fig2)

                # -------------------------
                # 圖表 3：單一方案雷達圖
                # -------------------------
                st.subheader("單一方案雷達圖")

                choice = st.selectbox("選擇要查看雷達圖的方案", result.X.index.tolist())

                values = result.X.loc[choice].values.astype(float)
                labels = result.X.columns.tolist()

                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                values = np.concatenate((values, [values[0]]))
                angles += angles[:1]

                fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax3.plot(angles, values, linewidth=2)
                ax3.fill(angles, values, alpha=0.25)
                ax3.set_xticks(angles[:-1])
                ax3.set_xticklabels(labels)
                ax3.set_title(f"{choice} 指標雷達圖")
                st.pyplot(fig3)

                # -------------------------
                # 展開顯示矩陣
                # -------------------------
                with st.expander("原始矩陣 X"):
                    st.dataframe(result.X, use_container_width=True)

                with st.expander("正規化矩陣 R"):
                    st.dataframe(result.R, use_container_width=True)

                with st.expander("加權矩陣 V"):
                    st.dataframe(result.V, use_container_width=True)

                with st.expander("理想解 A+ / 負理想解 A-"):
                    ideal_df = pd.DataFrame({
                        "A+": result.ideal_best,
                        "A-": result.ideal_worst
                    })
                    st.dataframe(ideal_df, use_container_width=True)

            except Exception as e:
                st.error(f"錯誤：{e}")

else:
    st.info("請先上傳 CSV 或 Excel 檔。")