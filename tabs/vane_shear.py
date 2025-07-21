import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import math

def run():
    st.subheader("🌀 Vane Shear Test (IS 2720 Part 30:1980)") # Ref: IS 2720 Part 30:1980
    st.markdown("""
    This test determines the **undrained shear strength ($S_u$)** of soft cohesive soils
    by measuring the torque required to rotate a vane within the soil.
    It can also be used to evaluate **Sensitivity**.

    **Shear Strength (S) formula:**
    $$
    S = \\frac{T}{K_v} \\quad \\text{where} \\quad K_v = \\pi D^2 H \\left( \\frac{1}{2} + \\frac{D}{6H} \\right)
    $$
    - $T$: Torque (kg-cm)
    - $K_v$: Vane constant (cm³)
    - $D$: Diameter of vane (cm)
    - $H$: Height of vane (cm)
    """)

    # --- Session State Initialization for Inputs ---
    if "vs_vane_D" not in st.session_state:
        st.session_state.vs_vane_D = 1.2 # cm (12mm)
    if "vs_vane_H" not in st.session_state:
        st.session_state.vs_vane_H = 2.0 # cm (20mm)
    if "vs_spring_constant" not in st.session_state:
        st.session_state.vs_spring_constant = 1.0 # kg-cm/degree (example)
    if "vs_num_trials" not in st.session_state:
        st.session_state.vs_num_trials = 2 # For undisturbed and remoulded

    # Initialize data for each trial
    if "vs_trials_data" not in st.session_state:
        st.session_state.vs_trials_data = []

    # --- General Parameters ---
    st.markdown("### 📏 Vane Dimensions & Apparatus Constant")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.vs_vane_D = st.number_input(
            "Vane Diameter (D) in cm",
            min_value=0.1, value=st.session_state.vs_vane_D, format="%.2f", key="vs_vane_D_input"
        )
        st.session_state.vs_vane_H = st.number_input(
            "Vane Height (H) in cm",
            min_value=0.1, value=st.session_state.vs_vane_H, format="%.2f", key="vs_vane_H_input"
        )
    with col2:
        st.session_state.vs_spring_constant = st.number_input(
            "Spring Constant (kg-cm/degree)",
            min_value=0.001, value=st.session_state.vs_spring_constant, format="%.3f", key="vs_spring_const_input"
        )
        # Calculate Vane Constant (Kv)
        if st.session_state.vs_vane_H > 0:
            Kv = (math.pi * (st.session_state.vs_vane_D**2) * st.session_state.vs_vane_H / 2) + \
                 (math.pi * (st.session_state.vs_vane_D**3) / 6)
            st.info(f"Calculated Vane Constant ($K_v$): **{Kv:.3f} cm³**")
        else:
            Kv = 0.0
            st.warning("Vane Height must be > 0 to calculate Vane Constant.")

    # --- Number of Trials ---
    num_trials = st.number_input(
        "Number of Trials (e.g., 1 for Undisturbed, 2 for Undisturbed & Remoulded)",
        min_value=1, max_value=3,
        value=st.session_state.vs_num_trials,
        step=1, key="vs_num_trials_input"
    )

    # Reinitialize/resize trials_data based on num_trials
    if num_trials != st.session_state.vs_num_trials:
        st.session_state.vs_num_trials = num_trials
        st.session_state.vs_trials_data = [] # Clear existing data if count changes
        # Ensure a template is available for new trials
        for i in range(num_trials):
            st.session_state.vs_trials_data.append({
                "Type": "Undisturbed" if i == 0 else "Remoulded",
                "Initial Reading (Deg)": 0.0,
                "Final Reading (Deg)": 0.0
            })
    
    # Initialize vs_trials_data if it's empty on first run
    if not st.session_state.vs_trials_data:
        for i in range(num_trials):
            st.session_state.vs_trials_data.append({
                "Type": "Undisturbed" if i == 0 else "Remoulded",
                "Initial Reading (Deg)": 0.0,
                "Final Reading (Deg)": 0.0
            })
    # Ensure list size matches current num_trials
    st.session_state.vs_trials_data = st.session_state.vs_trials_data[:num_trials]


    # --- Trial Data Input ---
    st.markdown("### 📋 Angle of Twist Readings")
    for i in range(num_trials):
        st.markdown(f"#### Trial {i+1} ({st.session_state.vs_trials_data[i]['Type']})")
        col_type, col_readings = st.columns([1, 2])
        with col_type:
            st.session_state.vs_trials_data[i]["Type"] = st.selectbox(
                f"Sample Type - Trial {i+1}",
                options=["Undisturbed", "Remoulded"],
                index=0 if i == 0 else 1, # Default to Undisturbed for first, Remoulded for others
                key=f"vs_sample_type_{i}"
            )
        with col_readings:
            st.session_state.vs_trials_data[i]["Initial Reading (Deg)"] = st.number_input(
                f"Initial Reading (Degrees) [{i+1}]",
                min_value=0.0, value=st.session_state.vs_trials_data[i]["Initial Reading (Deg)"],
                key=f"vs_initial_deg_{i}"
            )
            st.session_state.vs_trials_data[i]["Final Reading (Deg)"] = st.number_input(
                f"Final Reading (Degrees) [{i+1}]",
                min_value=0.0, value=st.session_state.vs_trials_data[i]["Final Reading (Deg)"],
                key=f"vs_final_deg_{i}"
            )

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_vs_inputs_button"):
        input_data_for_save = {
            "Vane Diameter (cm)": st.session_state.vs_vane_D,
            "Vane Height (cm)": st.session_state.vs_vane_H,
            "Spring Constant (kg-cm/degree)": st.session_state.vs_spring_constant,
            "Trial Data": st.session_state.vs_trials_data
        }
        
        buffer = StringIO()
        buffer.write("--- General Parameters ---\n")
        buffer.write(f"Vane Diameter (cm),{input_data_for_save['Vane Diameter (cm)']}\n")
        buffer.write(f"Vane Height (cm),{input_data_for_save['Vane Height (cm)']}\n")
        buffer.write(f"Spring Constant (kg-cm/degree),{input_data_for_save['Spring Constant (kg-cm/degree)']}\n")
        buffer.write("\n--- Trial Data ---\n")
        pd.DataFrame(input_data_for_save["Trial Data"]).to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="vane_shear_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Shear Strength Button ---
    if st.button("Calculate Shear Strength", key="calculate_vs_results_button"):
        calculated_trials = []
        undisturbed_strength = None
        remoulded_strength = None

        if Kv <= 0:
            st.error("Calculated Vane Constant is zero or negative. Check Vane Diameter and Height.")
            return None
        if st.session_state.vs_spring_constant <= 0:
            st.error("Spring Constant must be greater than zero.")
            return None

        for i, trial_input in enumerate(st.session_state.vs_trials_data):
            initial_deg = trial_input["Initial Reading (Deg)"]
            final_deg = trial_input["Final Reading (Deg)"]
            sample_type = trial_input["Type"]

            if final_deg < initial_deg:
                st.warning(f"Trial {i+1} ({sample_type}): Final reading is less than initial reading. Skipping calculation for this trial.")
                continue

            try:
                # Difference in degrees
                diff_deg = final_deg - initial_deg
                # Torque T (kg-cm)
                T = st.session_state.vs_spring_constant * diff_deg
                # Shear Strength S (kg/cm²)
                S = T / Kv

                calculated_trials.append({
                    "Trial": i + 1,
                    "Type": sample_type,
                    "Initial Reading (Deg)": initial_deg,
                    "Final Reading (Deg)": final_deg,
                    "Difference (Deg)": round(diff_deg, 2),
                    "Torque (kg-cm)": round(T, 3),
                    "Shear Strength (kg/cm²)": round(S, 3)
                })

                if sample_type == "Undisturbed":
                    undisturbed_strength = S
                elif sample_type == "Remoulded":
                    remoulded_strength = S

            except Exception as e:
                st.error(f"Trial {i+1} ({sample_type}): An error occurred during calculation: {e}. Check inputs.")
                continue

        if not calculated_trials:
            st.error("No valid trials processed. Please ensure all inputs are correctly filled.")
            return None

        results_df = pd.DataFrame(calculated_trials)
        st.markdown("### 📋 Shear Strength Calculation Results")
        st.dataframe(results_df, use_container_width=True)

        avg_shear_strength = results_df["Shear Strength (kg/cm²)"].mean()
        st.success(f"**Average Shear Strength (S)**: {avg_shear_strength:.3f} kg/cm²")

        # --- Sensitivity Calculation ---
        sensitivity = "N/A"
        if undisturbed_strength is not None and remoulded_strength is not None and remoulded_strength > 0:
            sensitivity = undisturbed_strength / remoulded_strength
            st.markdown("### 🧪 Sensitivity")
            st.info(f"**Sensitivity (S_t)** = Undisturbed Strength / Remoulded Strength = {sensitivity:.2f}")

            # Interpret Sensitivity
            if sensitivity < 1:
                st.warning("Sensitivity < 1: Indicates remoulded strength is higher than undisturbed, which is unusual. Check inputs.")
            elif 1 <= sensitivity <= 2:
                st.info("Sensitivity: **Insensitive**")
            elif 2 < sensitivity <= 4:
                st.info("Sensitivity: **Normal Sensitive**")
            elif 4 < sensitivity <= 8:
                st.warning("Sensitivity: **Sensitive**")
            elif 8 < sensitivity <= 16:
                st.error("Sensitivity: **Extra Sensitive**")
            else:
                st.error("Sensitivity: **Quick Clay**")
        else:
            st.info("To calculate Sensitivity, perform at least one Undisturbed and one Remoulded test.")
            sensitivity = "Requires Undisturbed and Remoulded strengths"

        # --- General Remarks ---
        st.markdown("### 📝 General Remarks")
        remarks = """
        1. The Vane Shear Test is highly suitable for soft, cohesive soils where undisturbed samples are difficult to obtain or handle.
        2. It provides the undrained shear strength directly.
        3. The test is relatively quick and economical.
        4. Sensitivity is a crucial parameter for evaluating the loss of strength upon remoulding, especially important for clays.
        """
        st.info(remarks)

        # Return results for the main app to collect
        return {
            "General Parameters": pd.DataFrame({
                "Parameter": ["Vane Diameter (cm)", "Vane Height (cm)", "Spring Constant (kg-cm/degree)", "Calculated Vane Constant (cm³)"],
                "Value": [st.session_state.vs_vane_D, st.session_state.vs_vane_H, st.session_state.vs_spring_constant, round(Kv, 3)]
            }),
            "Raw Trial Inputs": pd.DataFrame(st.session_state.vs_trials_data).insert(0, "Trial No.", range(1, len(st.session_state.vs_trials_data) + 1)),
            "Calculated Shear Strengths": results_df,
            "Average Shear Strength (kg/cm²)": f"{avg_shear_strength:.3f}",
            "Sensitivity (S_t)": str(sensitivity),
            "Remarks": remarks
        }

    return None # Default return if calculation button is not pressed