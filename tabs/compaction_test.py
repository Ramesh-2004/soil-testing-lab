import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compaction_test():
    st.subheader("Standard Proctor Compaction Test")
    st.markdown("""
        **Reference:** IS 2720: Part 7: 1980, IS 10074:1982, IS 9198:1978
        
        Enter the weight and volume data to compute Moisture Content vs Dry Density curve.
    """)

    with st.form("compaction_form"):
        data = st.text_area("Enter data as CSV: Water Content (%), Wet Density (g/cc)",
                            "10,1.6\n12,1.72\n14,1.78\n16,1.76\n18,1.72")
        submitted = st.form_submit_button("Compute")

    if submitted:
        try:
            df = pd.read_csv(pd.compat.StringIO(data), header=None, names=["Water Content", "Wet Density"])
            df["Dry Density"] = df["Wet Density"] / (1 + df["Water Content"] / 100)

            opt_idx = df["Dry Density"].idxmax()
            opt_w = df.loc[opt_idx, "Water Content"]
            opt_dry = df.loc[opt_idx, "Dry Density"]

            st.success(f"Optimum Moisture Content (OMC): {opt_w:.2f}%")
            st.success(f"Maximum Dry Density (MDD): {opt_dry:.2f} g/cc")

            fig, ax = plt.subplots()
            ax.plot(df["Water Content"], df["Dry Density"], marker='o', linestyle='-')
            ax.set_xlabel("Water Content (%)")
            ax.set_ylabel("Dry Density (g/cc)")
            ax.set_title("Compaction Curve")
            ax.grid(True)
            st.pyplot(fig)

            st.dataframe(df.style.format({"Water Content": "{:.2f}", "Wet Density": "{:.2f}", "Dry Density": "{:.2f}"}))

            st.markdown("""
            ### Suitability:
            - Soil with higher MDD and lower OMC is generally suitable for construction.
            - Compare with typical values for clay, silt, or sand.
            """)

        except Exception as e:
            st.error(f"Error in input or calculation: {e}")
