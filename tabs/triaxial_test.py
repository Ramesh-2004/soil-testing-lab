import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import math

def run():
    st.subheader("🚧 Undrained Triaxial Test (IS 2720 Part 11:1971)") # Ref: IS 2720 Part 11:1971
    st.markdown("""
    This test is performed to determine the **shear strength parameters (cohesion 'c' and
    angle of internal friction 'phi')** of cohesive soil samples under undrained conditions.
    Multiple soil samples are tested under different confining (cell) pressures.
    """)

    # --- Session State Initialization for Inputs ---
    # Use distinct keys for each parameter
    if "tri_num_samples" not in st.session_state:
        st.session_state.tri_num_samples = 3 # Number of samples/cell pressures
    if "tri_sample_dia" not in st.session_state:
        st.session_state.tri_sample_dia = 3.8 # cm (38mm)
    if "tri_sample_height" not in st.session_state:
        st.session_state.tri_sample_height = 7.6 # cm (76mm)
    if "tri_proving_ring_constant" not in st.session_state:
        st.session_state.tri_proving_ring_constant = 1.0 # kg/division (example)
    if "tri_deformation_lc" not in st.session_state:
        st.session_state.tri_deformation_lc = 0.01 # mm

    # Initialize a list in session state to hold dictionaries for each sample
    # Each dictionary will hold 'cell_pressure' and a unique key for its 'load_readings_df'
    if "tri_sample_configs" not in st.session_state:
        st.session_state.tri_sample_configs = []
    
    # --- General Parameters ---
    st.markdown("### 🔬 General Sample & Apparatus Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.tri_sample_dia = st.number_input(
            "Initial Diameter of Sample (cm)",
            min_value=0.1, value=st.session_state.tri_sample_dia, key="tri_dia_input"
        )
        st.session_state.tri_sample_height = st.number_input(
            "Initial Height of Sample (cm)",
            min_value=0.1, value=st.session_state.tri_sample_height, key="tri_height_input"
        )
    with col2:
        st.session_state.tri_proving_ring_constant = st.number_input(
            "Proving Ring Constant (kg/division)",
            min_value=0.01, value=st.session_state.tri_proving_ring_constant, format="%.2f", key="tri_pr_const_input"
        )
        st.session_state.tri_deformation_lc = st.number_input(
            "Deformation Dial Gauge Least Count (mm)",
            min_value=0.001, value=st.session_state.tri_deformation_lc, format="%.3f", key="tri_def_lc_input"
        )
    
    initial_area_cm2 = (math.pi / 4) * (st.session_state.tri_sample_dia ** 2)
    st.info(f"Calculated Initial Area of Sample: {initial_area_cm2:.2f} cm²")

    # --- Number of Samples/Cell Pressures ---
    num_samples = st.number_input(
        "Number of Samples / Cell Pressures to Test (Min 2 for c & phi)",
        min_value=1, max_value=5,
        value=st.session_state.tri_num_samples,
        step=1, key="tri_num_samples_input"
    )

    # Adjust tri_sample_configs based on num_samples
    if num_samples != len(st.session_state.tri_sample_configs):
        # If increasing samples, append new default configs
        while len(st.session_state.tri_sample_configs) < num_samples:
            new_sample_index = len(st.session_state.tri_sample_configs)
            default_df = pd.DataFrame({
                "Deformation Dial Reading (Div)": np.arange(0, 401, 50),
                "Proving Ring Reading (Div)": [0.0] * len(np.arange(0, 401, 50))
            })
            st.session_state.tri_sample_configs.append({
                "cell_pressure": (new_sample_index + 1) * 0.5,
                f"load_readings_df_{new_sample_index}": default_df # Use a unique key for the DataFrame itself
            })
        # If decreasing samples, truncate the list
        st.session_state.tri_sample_configs = st.session_state.tri_sample_configs[:num_samples]
        st.session_state.tri_num_samples = num_samples # Update stored num_samples

    # --- Input Fields for Each Sample/Cell Pressure ---
    for i in range(st.session_state.tri_num_samples):
        st.markdown(f"### Sample {i+1}")
        
        # Input for Cell Pressure
        # Update the dictionary in tri_sample_configs directly
        st.session_state.tri_sample_configs[i]["cell_pressure"] = st.number_input(
            f"Cell Pressure ($\sigma_3$) (kg/cm²) - Sample {i+1}",
            min_value=0.0, value=st.session_state.tri_sample_configs[i]["cell_pressure"], format="%.2f",
            key=f"tri_cell_pressure_{i}"
        )
        
        st.markdown(f"**Load & Deformation Readings for Sample {i+1}**")
        
        # Retrieve the DataFrame using its unique key for this sample
        load_readings_df_key = f"load_readings_df_{i}"
        current_load_df = st.session_state.tri_sample_configs[i][load_readings_df_key]
        
        edited_current_load_df = st.data_editor(
            current_load_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"tri_readings_editor_{i}" # Unique key for each data_editor
        )
        
        # Crucial: Update the specific DataFrame within the sample's config
        st.session_state.tri_sample_configs[i][load_readings_df_key] = edited_current_load_df


    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_tri_inputs_button"):
        buffer = StringIO()
        
        buffer.write("--- General Parameters ---\n")
        buffer.write(f"Initial Sample Diameter (cm),{st.session_state.tri_sample_dia}\n")
        buffer.write(f"Initial Sample Height (cm),{st.session_state.tri_sample_height}\n")
        buffer.write(f"Proving Ring Constant (kg/division),{st.session_state.tri_proving_ring_constant}\n")
        buffer.write(f"Deformation Dial Gauge Least Count (mm),{st.session_state.tri_deformation_lc}\n")
        buffer.write(f"Calculated Initial Area (cm²),{initial_area_cm2:.2f}\n")
        
        for i, sample_config in enumerate(st.session_state.tri_sample_configs):
            buffer.write(f"\n--- Sample {i+1} ---\n")
            buffer.write(f"Cell Pressure (kg/cm²),{sample_config['cell_pressure']}\n")
            buffer.write("--- Load & Deformation Readings ---\n")
            # Access the DataFrame using its dynamic key
            sample_config[f"load_readings_df_{i}"].to_csv(buffer, index=False)
        
        buffer.seek(0)
        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="triaxial_test_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Triaxial Results Button ---
    if st.button("Calculate Triaxial Results", key="calculate_tri_results_button"):
        all_sample_results = [] # To store calculated data for each sample
        failure_points = [] # For Mohr's Circle plot (sigma1, sigma3)

        # General parameters from session state
        initial_h = st.session_state.tri_sample_height
        pr_constant = st.session_state.tri_proving_ring_constant
        def_lc = st.session_state.tri_deformation_lc

        if not (initial_h > 0 and pr_constant > 0 and def_lc > 0):
            st.error("Please ensure initial sample height, proving ring constant, and deformation dial gauge least count are positive.")
            return None

        # Loop through each sample/cell pressure
        for i, sample_config in enumerate(st.session_state.tri_sample_configs):
            cell_pressure = sample_config["cell_pressure"]
            # Retrieve the correct DataFrame for calculation
            readings_df = sample_config[f"load_readings_df_{i}"].copy()

            if readings_df.empty:
                st.warning(f"Sample {i+1} (Cell Pressure: {cell_pressure} kg/cm²): No readings entered. Skipping.")
                continue

            # Calculate Deformation (mm to cm)
            readings_df["Deformation (cm)"] = readings_df["Deformation Dial Reading"] * def_lc / 10

            # Calculate Strain (cm/cm, then convert to %)
            readings_df["Strain (%)"] = (readings_df["Deformation (cm)"] / initial_h) * 100

            # Calculate Axial Load (kg)
            readings_df["Axial Load (kg)"] = readings_df["Proving Ring Reading"] * pr_constant

            # Calculate Corrected Area (Ac)
            # Ac = A0 / (1 - ε_decimal)
            readings_df["Corrected Area (cm²)"] = initial_area_cm2 / (1 - readings_df["Strain (%)"] / 100)
            readings_df["Corrected Area (cm²)"] = readings_df["Corrected Area (cm²)"].replace([np.inf, -np.inf], np.nan)

            # Calculate Deviator Stress (σ_d = Load / Corrected Area)
            readings_df["Deviator Stress (kg/cm²)"] = readings_df["Axial Load (kg)"] / readings_df["Corrected Area (cm²)"]
            
            # Drop rows with NaN in critical columns
            readings_df = readings_df.dropna(subset=["Strain (%)", "Deviator Stress (kg/cm²)", "Corrected Area (cm²)"]).copy()

            if readings_df.empty:
                st.warning(f"Sample {i+1} (Cell Pressure: {cell_pressure} kg/cm²): No valid stress-strain data after filtering. Skipping.")
                continue

            # Find maximum Deviator Stress (Failure Point)
            peak_deviator_stress = readings_df["Deviator Stress (kg/cm²)"].max()
            
            # Major Principal Stress (sigma1) = Cell Pressure + Deviator Stress at failure
            sigma1 = cell_pressure + peak_deviator_stress
            # Minor Principal Stress (sigma3) = Cell Pressure
            sigma3 = cell_pressure

            failure_points.append({"sigma1": sigma1, "sigma3": sigma3})
            
            # Store results for this sample
            all_sample_results.append({
                "Sample Number": i + 1,
                "Cell Pressure (kg/cm²)": cell_pressure,
                "Calculated Data": readings_df, # Full DataFrame for this sample
                "Peak Deviator Stress (kg/cm²)": round(peak_deviator_stress, 3),
                "Sigma1 at Failure (kg/cm²)": round(sigma1, 3),
                "Sigma3 at Failure (kg/cm²)": round(sigma3, 3)
            })

            # Display individual sample results
            st.markdown(f"#### Sample {i+1} Results (Cell Pressure: {cell_pressure} kg/cm²)")
            st.dataframe(readings_df[['Deformation (cm)', 'Strain (%)', 'Axial Load (kg)', 'Corrected Area (cm²)', 'Deviator Stress (kg/cm²)']].round(3), use_container_width=True)
            st.success(f"Peak Deviator Stress ($\sigma_d$): {peak_deviator_stress:.3f} kg/cm²")
            st.info(f"Principal Stresses at Failure: $\\sigma_1 = {sigma1:.3f}$ kg/cm², $\\sigma_3 = {sigma3:.3f}$ kg/cm²")
        
        if not all_sample_results or len(failure_points) < 2:
            st.error("Not enough valid samples processed (minimum 2 required) to determine shear strength parameters (c & phi).")
            return None

        # --- Plot Stress vs. Strain Curves for all samples ---
        st.markdown("### 📈 Deviator Stress vs. Strain Curves")
        fig_stress_strain, ax_stress_strain = plt.subplots(figsize=(10, 6))

        for res in all_sample_results:
            sample_num = res["Sample Number"]
            cell_p = res["Cell Pressure (kg/cm²)"]
            data_df = res["Calculated Data"]
            ax_stress_strain.plot(data_df["Strain (%)"], data_df["Deviator Stress (kg/cm²)"], marker='.', linestyle='-', label=f'Sample {sample_num} ($\sigma_3={cell_p}$ kg/cm²)')
        
        ax_stress_strain.set_xlabel("Axial Strain (%)")
        ax_stress_strain.set_ylabel("Deviator Stress ($\sigma_d$) kg/cm²")
        ax_stress_strain.set_title("Triaxial Test: Deviator Stress vs. Strain Curves")
        ax_stress_strain.grid(True, which="both", ls="--", color='0.7')
        ax_stress_strain.legend()
        st.pyplot(fig_stress_strain)

        img_buf_stress_strain = BytesIO()
        fig_stress_strain.savefig(img_buf_stress_strain, format="png", bbox_inches="tight")
        img_buf_stress_strain.seek(0)
        plt.close(fig_stress_strain) # Close figure to free memory

        # --- Plot Mohr's Circles and Failure Envelope ---
        st.markdown("### ⚪ Mohr's Circles and Failure Envelope")
        fig_mohr, ax_mohr = plt.subplots(figsize=(10, 8))
        
        centers = []
        radii = []

        for fp in failure_points:
            center = (fp["sigma1"] + fp["sigma3"]) / 2
            radius = (fp["sigma1"] - fp["sigma3"]) / 2
            centers.append(center)
            radii.append(radius)
            
            circle = plt.Circle((center, 0), radius, color='blue', fill=False, linewidth=1.5, label=f'$\sigma_3$={fp["sigma3"]}')
            ax_mohr.add_artist(circle)
            ax_mohr.plot(center + radius, 0, 'rx') # Mark sigma1 on x-axis
            ax_mohr.plot(center - radius, 0, 'rx') # Mark sigma3 on x-axis

        ax_mohr.set_xlabel("Normal Stress ($\sigma$) kg/cm²")
        ax_mohr.set_ylabel("Shear Stress ($\\tau$) kg/cm²")
        ax_mohr.set_title("Mohr-Coulomb Failure Envelope")
        ax_mohr.set_aspect('equal', adjustable='box') # Keep circles circular
        ax_mohr.grid(True, which="both", ls="--", color='0.7')
        ax_mohr.axhline(0, color='black', linewidth=0.5)
        ax_mohr.axvline(0, color='black', linewidth=0.5)

        # Fit failure envelope (tangent to Mohr circles)
        if len(centers) >= 2:
            c_u_values = [r for r in radii] # For UU, c_u is the radius (deviator stress at failure/2 for UCS, but here it's (sigma1-sigma3)/2)
            average_c_u = np.mean(c_u_values) if c_u_values else 0

            st.success(f"**Average Undrained Shear Strength ($c_u$)**: {average_c_u:.3f} kg/cm²")
            st.info("For Undrained Triaxial Test (UU), the angle of internal friction ($\phi_u$) is typically considered 0 for saturated cohesive soils.")
            
            # Draw a horizontal line for phi_u=0 envelope
            ax_mohr.axhline(average_c_u, color='red', linestyle='--', label=f'Failure Envelope ($\\tau={average_c_u:.3f}$ kg/cm²)')
            cohesion_val = round(average_c_u, 3)
            phi_val = 0.0 # Assumed for UU test
            
        else: # Less than 2 failure points
            cohesion_val = "N/A (Need >= 2 samples)"
            phi_val = "N/A (Need >= 2 samples)"
            st.warning("Cannot determine cohesion (c) and angle of internal friction (phi) without at least two valid test samples.")

        ax_mohr.legend()
        st.pyplot(fig_mohr)

        img_buf_mohr = BytesIO()
        fig_mohr.savefig(img_buf_mohr, format="png", bbox_inches="tight")
        img_buf_mohr.seek(0)
        plt.close(fig_mohr)


        # --- Soil Consistency from c_u ---
        st.markdown("### 📝 Consistency of Cohesive Soil (Based on $c_u$)")
        consistency = ""
        if isinstance(cohesion_val, (int, float)):
            if cohesion_val < 0.25:
                consistency = "Very Soft"
            elif 0.25 <= cohesion_val < 0.50:
                consistency = "Soft"
            elif 0.50 <= cohesion_val < 1.00:
                consistency = "Medium"
            elif 1.00 <= cohesion_val < 2.00:
                consistency = "Stiff"
            elif 2.00 <= cohesion_val < 4.00:
                consistency = "Very Stiff"
            else:
                consistency = "Hard"
            st.info(f"The soil consistency based on $c_u$ is **{consistency}**.")
        else:
            consistency = "Could not determine (c_u not calculated)."
            st.info(consistency)


        # Return results for the main app to collect
        return {
            "General Parameters": pd.DataFrame({
                "Parameter": ["Initial Diameter (cm)", "Initial Height (cm)", "Proving Ring Constant (kg/div)", "Deformation Dial LC (mm)", "Initial Area (cm²)"],
                "Value": [st.session_state.tri_sample_dia, st.session_state.tri_sample_height, st.session_state.tri_proving_ring_constant, st.session_state.tri_deformation_lc, round(initial_area_cm2, 2)]
            }),
            # Updated to reflect new session state structure
            "Raw Trial Inputs": pd.DataFrame([{"Sample": i+1, "Cell Pressure (kg/cm²)": s["cell_pressure"],"Load Readings": s[f"load_readings_df_{i}"].to_dict('records')} for i, s in enumerate(st.session_state.tri_sample_configs)]),
            "Sample Failure Data": pd.DataFrame([{"Sample": r["Sample Number"], "Cell Pressure (kg/cm²)": r["Cell Pressure (kg/cm²)"], "Peak Deviator Stress (kg/cm²)": r["Peak Deviator Stress (kg/cm²)"], "Sigma1 (kg/cm²)": r["Sigma1 at Failure (kg/cm²)"], "Sigma3 (kg/cm²)": r["Sigma3 at Failure (kg/cm²)"]} for r in all_sample_results]),
            "Deviator Stress-Strain Curves": img_buf_stress_strain,
            "Mohr's Circles & Failure Envelope": img_buf_mohr,
            "Cohesion (c)": f"{cohesion_val} kg/cm²",
            "Angle of Internal Friction (phi)": f"{phi_val} degrees",
            "Soil Consistency": consistency,
            "Remarks": "Shear strength parameters (c, phi) determined using the Undrained Triaxial Test. Note that for truly Undrained (UU) tests on saturated clays, phi is often assumed to be zero."
        }
    
    return None # Default return if calculation button is not pressed

# If you're running this as a standalone script:
if __name__ == "__main__":
    run()