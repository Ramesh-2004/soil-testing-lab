import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import math

def run():
    st.subheader("💪 Unconfined Compressive Strength (UCS) Test (IS 2720 Part 10:1991)")
    st.markdown("""
    This test is performed to determine the **Unconfined Compressive Strength ($\sigma_c$)** and
    **Undrained Shear Strength ($c_u$)** of cohesive soil samples. It is suitable for soils
    that can stand without lateral support.
    """)

    # --- Session State Initialization for Inputs ---
    # Fix applied here: Ensure citation is inside the comment.
    [cite_start]st.session_state.ucs_sample_dia = 3.8 # cm [cite: 5]
    if "ucs_sample_dia" not in st.session_state:
        [cite_start]st.session_state.ucs_sample_dia = 3.8 # cm [cite: 5]

    [cite_start]st.session_state.ucs_sample_height = 7.6 # cm [cite: 5]
    if "ucs_sample_height" not in st.session_state:
        [cite_start]st.session_state.ucs_sample_height = 7.6 # cm [cite: 5]

    if "ucs_proving_ring_constant" not in st.session_state:
        st.session_state.ucs_proving_ring_constant = 1.0 # kg/division (example)
    if "ucs_initial_weight" not in st.session_state:
        st.session_state.ucs_initial_weight = 0.0 # g
    if "ucs_moisture_content" not in st.session_state:
        st.session_state.ucs_moisture_content = 0.0 # % (for calculating initial dry density if needed)


    if "ucs_load_deformation_data" not in st.session_state:
        st.session_state.ucs_load_deformation_data = pd.DataFrame({
            "Deformation Dial Reading": np.arange(0, 10.5, 0.5), # mm
            "Proving Ring Reading": [0.0] * len(np.arange(0, 10.5, 0.5))
        })

    # --- Sample Dimensions & Initial Data ---
    st.markdown("### 🔬 Sample Dimensions & Initial Data")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.ucs_sample_dia = st.number_input(
            "Initial Diameter of Sample (cm)",
            min_value=0.1, value=st.session_state.ucs_sample_dia, key="ucs_dia_input"
        )
        st.session_state.ucs_sample_height = st.number_input(
            "Initial Height of Sample (cm)",
            min_value=0.1, value=st.session_state.ucs_sample_height, key="ucs_height_input"
        )
    with col2:
        st.session_state.ucs_proving_ring_constant = st.number_input(
            "Proving Ring Constant (kg/division)",
            min_value=0.01, value=st.session_state.ucs_proving_ring_constant, format="%.2f", key="ucs_pr_const_input"
        )
        st.session_state.ucs_initial_weight = st.number_input(
            "Initial Weight of Sample (g)",
            min_value=0.0, value=st.session_state.ucs_initial_weight, key="ucs_initial_weight_input"
        )
        st.session_state.ucs_moisture_content = st.number_input(
            "Moisture Content of Sample (%) (Optional)",
            min_value=0.0, value=st.session_state.ucs_moisture_content, key="ucs_mc_input"
        )

    initial_area = (math.pi / 4) * (st.session_state.ucs_sample_dia ** 2)
    st.info(f"Calculated Initial Area of Sample: {initial_area:.2f} cm²")

    # --- Load Deformation Data Input ---
    st.markdown("### 📋 Load & Deformation Readings")
    st.markdown("Enter Proving Ring Readings for corresponding Deformation Dial Readings (mm).")

    edited_ucs_df = st.data_editor(
        st.session_state.ucs_load_deformation_data,
        num_rows="dynamic", # Allow adding more points if needed
        use_container_width=True,
        key="ucs_data_editor"
    )
    st.session_state.ucs_load_deformation_data = edited_ucs_df # Update session state

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_ucs_inputs_button"):
        input_data_for_save = {
            "Initial Sample Diameter (cm)": st.session_state.ucs_sample_dia,
            "Initial Sample Height (cm)": st.session_state.ucs_sample_height,
            "Proving Ring Constant (kg/division)": st.session_state.ucs_proving_ring_constant,
            "Initial Weight of Sample (g)": st.session_state.ucs_initial_weight,
            "Moisture Content of Sample (%)": st.session_state.ucs_moisture_content,
            "Load Deformation Data": st.session_state.ucs_load_deformation_data.to_dict('records')
        }

        # Convert dictionary to a string for simple CSV saving
        buffer = StringIO()
        buffer.write("--- General Parameters ---\n")
        for key, value in input_data_for_save.items():
            if not isinstance(value, list): # Exclude the load_deformation_data list for now
                buffer.write(f"{key},{value}\n")
        buffer.write("\n--- Load & Deformation Data ---\n")
        st.session_state.ucs_load_deformation_data.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="ucs_test_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate UCS Button ---
    if st.button("Calculate UCS", key="calculate_ucs_button"):
        df = st.session_state.ucs_load_deformation_data.copy()

        # Validate general parameters
        if not (st.session_state.ucs_sample_dia > 0 and st.session_state.ucs_sample_height > 0 and
                st.session_state.ucs_proving_ring_constant > 0):
            st.error("Please ensure Sample Diameter, Height, and Proving Ring Constant are positive.")
            return None

        # Calculate Load (kg)
        df["Load (kg)"] = df["Proving Ring Reading"] * st.session_state.ucs_proving_ring_constant

        # Calculate Deformation (cm)
        df["Deformation (cm)"] = df["Deformation Dial Reading"] / 10 # Convert mm to cm

        # Calculate Strain (ε)
        # Ensure initial_height is not zero to avoid division by zero
        if st.session_state.ucs_sample_height == 0:
            st.error("Initial sample height cannot be zero.")
            return None
        df["Strain (%)"] = (df["Deformation (cm)"] / st.session_state.ucs_sample_height) * 100

        # Calculate Corrected Area (Ac)
        # Ac = A0 / (1 - ε)
        # A0 is initial_area, ε is strain as a decimal (Strain % / 100)
        df["Corrected Area (cm²)"] = initial_area / (1 - df["Strain (%)"] / 100)

        # Handle cases where (1 - Strain/100) becomes zero or negative (i.e., strain >= 100%)
        # This can happen if deformation is too large or initial height is very small.
        # Replace inf/-inf with NaN for plotting robustness.
        df["Corrected Area (cm²)"] = df["Corrected Area (cm²)"].replace([np.inf, -np.inf], np.nan)


        # Calculate Stress (σ)
        df["Stress (kg/cm²)"] = df["Load (kg)"] / df["Corrected Area (cm²)"]

        # Drop rows with NaN in critical columns (e.g., from invalid area correction)
        df_results = df.dropna(subset=["Strain (%)", "Stress (kg/cm²)", "Corrected Area (cm²)"]).copy()

        if df_results.empty:
            st.error("No valid data points for stress-strain calculation after filtering. Please check your inputs and ensure deformation does not exceed initial height significantly.")
            return None

        st.markdown("### 📊 Calculated Stress-Strain Data")
        st.dataframe(df_results.round(3), use_container_width=True)

        # --- Plot Stress vs. Strain Curve ---
        st.markdown("### 📈 Stress vs. Strain Curve")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df_results["Strain (%)"], df_results["Stress (kg/cm²)"], marker='o', linestyle='-', color='red', label='Stress-Strain Curve')
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Stress (kg/cm²)")
        ax.set_title("Unconfined Compressive Strength Test")
        ax.grid(True, which="both", ls="--", color='0.7')
        ax.legend()

        # Find peak stress (Unconfined Compressive Strength)
        peak_stress_idx = df_results["Stress (kg/cm²)"].idxmax()
        ucs_strength = df_results.loc[peak_stress_idx, "Stress (kg/cm²)"]
        ucs_strain = df_results.loc[peak_stress_idx, "Strain (%)"]

        ax.plot(ucs_strain, ucs_strength, 'go', markersize=8, label=f'UCS = {ucs_strength:.2f} kg/cm²')
        ax.vlines(ucs_strain, ax.get_ylim()[0], ucs_strength, color='green', linestyle=':', linewidth=0.7)
        ax.hlines(ucs_strength, ax.get_xlim()[0], ucs_strain, color='green', linestyle=':', linewidth=0.7)
        ax.legend()

        st.pyplot(fig)

        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)
        plt.close(fig) # Close the figure to free memory

        # --- Results Summary ---
        st.markdown("### ✨ Test Results")
        st.success(f"**Unconfined Compressive Strength ($\sigma_c$)**: {ucs_strength:.2f} kg/cm²")

        # Undrained Shear Strength (cu) = sigma_c / 2
        cu = ucs_strength / 2
        st.success(f"**Undrained Shear Strength ($c_u$)**: {cu:.2f} kg/cm²")

        # --- Soil Consistency (from IS 1498:1970 Table 2, typical correlation) ---
        [cite_start]st.markdown("### 📝 Consistency of Cohesive Soil (IS 1498:1970 - Table 2)") # [cite: 7, 8]
        consistency = ""
        if ucs_strength < 0.25:
            consistency = "Very Soft"
        elif 0.25 <= ucs_strength < 0.50:
            consistency = "Soft"
        elif 0.50 <= ucs_strength < 1.00:
            consistency = "Medium"
        elif 1.00 <= ucs_strength < 2.00:
            consistency = "Stiff"
        elif 2.00 <= ucs_strength < 4.00:
            consistency = "Very Stiff"
        else:
            consistency = "Hard"
        st.info(f"The soil consistency is **{consistency}**.")

        # Return results for the main app to collect
        return {
            "Sample Dimensions & Initial Data": pd.DataFrame({
                "Parameter": ["Initial Diameter (cm)", "Initial Height (cm)", "Proving Ring Constant (kg/div)", "Deformation Dial LC (mm)", "Initial Weight (g)", "Moisture Content (%)", "Initial Area (cm²)"],
                "Value": [st.session_state.ucs_sample_dia, st.session_state.ucs_sample_height, st.session_state.ucs_proving_ring_constant, st.session_state.ucs_deformation_lc, st.session_state.ucs_initial_weight, st.session_state.ucs_moisture_content, round(initial_area, 2)]
            }),
            "Load Deformation Data (Raw)": st.session_state.ucs_load_deformation_data,
            "Calculated Stress-Strain Data": df_results,
            "Stress-Strain Curve": img_buf,
            "Unconfined Compressive Strength (sigma_c)": f"{ucs_strength:.2f} kg/cm²",
            "Undrained Shear Strength (c_u)": f"{cu:.2f} kg/cm²",
            "Soil Consistency": consistency,
            "Remarks": "Unconfined Compressive Strength and Undrained Shear Strength determined for cohesive soil."
        }

    return None # Default return if calculation button is not pressed