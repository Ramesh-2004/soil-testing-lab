import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO

def run():
    st.subheader("🛣️ California Bearing Ratio (CBR) Test (IS 2720 Part 16:1987)")
    st.markdown("""
    This test determines the **California Bearing Ratio (CBR)**, which is a measure of
    the strength of subgrade soils for roads and pavements.

    **CBR** = (Test Load / Standard Load) $\times$ 100
    """)

    # --- Standard Loads (from PDF Table 8.1) ---
    standard_loads_data = {
        "Penetration (mm)": [2.5, 5.0, 7.5, 10.0, 12.5],
        "Standard Load (kg)": [1370, 2055, 2630, 3180, 3600]
    }
    standard_loads_df = pd.DataFrame(standard_loads_data)

    st.markdown("### 📚 Standard Load Reference (IS 2720 Part 16)")
    st.dataframe(standard_loads_df, use_container_width=True)

    # --- Session State Initialization for Inputs ---
    if "cbr_mould_dia" not in st.session_state:
        st.session_state.cbr_mould_dia = 150.0 # mm
    if "cbr_mould_height" not in st.session_state:
        st.session_state.cbr_mould_height = 175.0 # mm
    if "cbr_proving_ring_constant" not in st.session_state:
        st.session_state.cbr_proving_ring_constant = 1.0 # kg/division (example)
    if "cbr_initial_penetration_offset" not in st.session_state:
        st.session_state.cbr_initial_penetration_offset = 0.0 # mm for curve correction

    if "cbr_load_penetration_data" not in st.session_state:
        st.session_state.cbr_load_penetration_data = pd.DataFrame({
            "Penetration (mm)": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5],
            "Proving Ring Reading": [0.0] * 12
        })

    # --- General Observations/Mould Data ---
    st.markdown("### 📏 General Observations & Equipment Data")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.cbr_mould_dia = st.number_input(
            "Mould Diameter (mm)",
            min_value=1.0, value=st.session_state.cbr_mould_dia, key="cbr_mould_dia_input"
        )
        st.session_state.cbr_mould_height = st.number_input(
            "Mould Height (mm)",
            min_value=1.0, value=st.session_state.cbr_mould_height, key="cbr_mould_height_input"
        )
    with col2:
        st.session_state.cbr_proving_ring_constant = st.number_input(
            "Proving Ring Constant (kg/division)",
            min_value=0.01, value=st.session_state.cbr_proving_ring_constant, format="%.2f", key="cbr_pr_const_input"
        )
        st.session_state.cbr_initial_penetration_offset = st.number_input(
            "Initial Penetration Offset for Curve Correction (mm)",
            min_value=-5.0, max_value=5.0, value=st.session_state.cbr_initial_penetration_offset, step=0.1, format="%.1f",
            help="If the initial part of the curve is concave upwards, enter the offset to shift the origin (e.g., 0.5 mm).",
            key="cbr_offset_input"
        )

    # --- Load Penetration Data Input ---
    st.markdown("### 📋 Load Penetration Readings")
    st.markdown("Enter Proving Ring Readings for each Penetration (mm).")

    edited_cbr_df = st.data_editor(
        st.session_state.cbr_load_penetration_data,
        num_rows="fixed", # Fixed rows as per standard penetrations
        use_container_width=True,
        key="cbr_data_editor"
    )
    st.session_state.cbr_load_penetration_data = edited_cbr_df # Update session state

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_cbr_inputs_button"):
        input_data_for_save = {
            "Mould Diameter (mm)": st.session_state.cbr_mould_dia,
            "Mould Height (mm)": st.session_state.cbr_mould_height,
            "Proving Ring Constant (kg/division)": st.session_state.cbr_proving_ring_constant,
            "Initial Penetration Offset (mm)": st.session_state.cbr_initial_penetration_offset,
            "Load Penetration Data": st.session_state.cbr_load_penetration_data.to_dict('records')
        }
        
        # Convert dictionary to a string for simple CSV saving
        buffer = StringIO()
        buffer.write("--- General Parameters ---\n")
        for key, value in input_data_for_save.items():
            if not isinstance(value, list): # Exclude the load_penetration_data list for now
                buffer.write(f"{key},{value}\n")
        buffer.write("\n--- Load Penetration Data ---\n")
        st.session_state.cbr_load_penetration_data.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="cbr_test_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate CBR Button ---
    if st.button("Calculate CBR", key="calculate_cbr_button"):
        df = st.session_state.cbr_load_penetration_data.copy()

        # Calculate Load (kg)
        df["Load (kg)"] = df["Proving Ring Reading"] * st.session_state.cbr_proving_ring_constant

        # Apply Penetration Correction
        df["Corrected Penetration (mm)"] = df["Penetration (mm)"] - st.session_state.cbr_initial_penetration_offset
        # Filter out negative corrected penetrations if offset is large
        df_plot = df[df["Corrected Penetration (mm)"] >= 0].copy()

        if df_plot.empty or len(df_plot) < 2:
            st.error("Not enough valid data points after applying offset for plotting. Please check inputs and offset.")
            return None

        # --- Plot Load vs. Penetration Curve ---
        st.markdown("### 📈 Load vs. Penetration Curve")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df_plot["Corrected Penetration (mm)"], df_plot["Load (kg)"], marker='o', linestyle='-', color='blue', label='Test Data')
        ax.set_xlabel("Penetration (mm)")
        ax.set_ylabel("Load (kg)")
        ax.set_title("CBR Load vs. Penetration Curve")
        ax.grid(True, which="both", ls="--", color='0.7')
        ax.legend()

        # Ensure x-axis starts from 0 for penetration
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        st.pyplot(fig)

        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)
        plt.close(fig) # Close the figure to free memory

        # --- Calculate CBR Values at 2.5mm and 5.0mm ---
        st.markdown("### 🔢 CBR Calculation")
        
        # Ensure data is sorted by corrected penetration for interpolation
        df_plot_sorted = df_plot.sort_values(by="Corrected Penetration (mm)").reset_index(drop=True)

        # Interpolate loads at 2.5mm and 5.0mm
        cbr_2_5_mm_load = np.interp(
            2.5,
            df_plot_sorted["Corrected Penetration (mm)"],
            df_plot_sorted["Load (kg)"]
        )
        cbr_5_0_mm_load = np.interp(
            5.0,
            df_plot_sorted["Corrected Penetration (mm)"],
            df_plot_sorted["Load (kg)"]
        )

        # Get standard loads
        std_load_2_5_mm = standard_loads_df[standard_loads_df["Penetration (mm)"] == 2.5]["Standard Load (kg)"].iloc[0]
        std_load_5_0_mm = standard_loads_df[standard_loads_df["Penetration (mm)"] == 5.0]["Standard Load (kg)"].iloc[0]

        # Calculate CBR
        cbr_2_5 = (cbr_2_5_mm_load / std_load_2_5_mm) * 100
        cbr_5_0 = (cbr_5_0_mm_load / std_load_5_0_mm) * 100

        st.success(f"**Load at 2.5 mm penetration**: {cbr_2_5_mm_load:.2f} kg")
        st.success(f"**CBR at 2.5 mm penetration**: {cbr_2_5:.2f}%")
        st.success(f"**Load at 5.0 mm penetration**: {cbr_5_0_mm_load:.2f} kg")
        st.success(f"**CBR at 5.0 mm penetration**: {cbr_5_0:.2f}%")

        # Final CBR for design
        final_cbr = max(cbr_2_5, cbr_5_0)
        cbr_remark = ""
        if cbr_2_5 >= cbr_5_0:
            final_cbr = cbr_2_5
            cbr_remark = "CBR at 2.5 mm penetration is greater or equal, taken for design."
        else:
            final_cbr = cbr_5_0
            cbr_remark = "CBR at 5.0 mm penetration is greater, taken for design. (Note: IS code suggests repeating test if 5mm > 2.5mm and if identical results follow, then 5mm CBR can be taken)."
            st.warning("According to IS 2720 Part 16, if CBR at 5.0 mm exceeds that for 2.5 mm, the test should be repeated. If identical results follow, the CBR corresponding to 5.0 mm penetration should be taken for design.")

        st.markdown(f"### ✨ **Final CBR for Design: {final_cbr:.2f}%**")
        st.info(cbr_remark)

        # --- Suitability Remarks ---
        st.markdown("### 📝 Suitability Remarks (General Guidance)")
        suitability_text = ""
        if final_cbr < 3:
            suitability_text = "Very poor subgrade. Requires significant improvement or thick pavement layers."
            st.error(suitability_text)
        elif 3 <= final_cbr < 7:
            suitability_text = "Poor to fair subgrade. Requires substantial pavement thickness."
            st.warning(suitability_text)
        elif 7 <= final_cbr < 20:
            suitability_text = "Fair to good subgrade. Suitable for moderate pavement thicknesses."
            st.info(suitability_text)
        else:
            suitability_text = "Excellent subgrade. Suitable for thin pavement layers or base course material."
            st.success(suitability_text)

        # Return results for the main app to collect
        return {
            "General Parameters": pd.DataFrame({
                "Parameter": ["Mould Diameter (mm)", "Mould Height (mm)", "Proving Ring Constant (kg/division)", "Initial Penetration Offset (mm)"],
                "Value": [st.session_state.cbr_mould_dia, st.session_state.cbr_mould_height, st.session_state.cbr_proving_ring_constant, st.session_state.cbr_initial_penetration_offset]
            }),
            "Raw Load-Penetration Data": df, # Includes calculated Load and Corrected Penetration
            "CBR Load-Penetration Curve": img_buf,
            "CBR at 2.5 mm (%)": f"{cbr_2_5:.2f}%",
            "CBR at 5.0 mm (%)": f"{cbr_5_0:.2f}%",
            "Final CBR for Design (%)": f"{final_cbr:.2f}%",
            "CBR Design Remark": cbr_remark,
            "Suitability Remarks": suitability_text,
            "Remarks": "California Bearing Ratio (CBR) determined for pavement design."
        }
    
    return None # Default return if calculation button is not pressed