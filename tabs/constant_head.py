import streamlit as st
import pandas as pd
from io import BytesIO, StringIO # Import StringIO for CSV buffer

# The main function that will be called by app.py
def run():
    st.subheader("Constant Head Permeability Test (IS 2720: Part 36: 1987)")
    st.markdown("This test is used to determine the coefficient of permeability for coarse-grained soils like sand and gravel.")

    # Initialize session state for the number of trials if not already set
    if "ch_num_trials" not in st.session_state:
        st.session_state.ch_num_trials = 3

    # Number of Trials input
    num_trials = st.number_input(
        "Number of Trials",
        min_value=1,
        max_value=10,
        value=st.session_state.ch_num_trials,
        step=1,
        key="ch_num_trials_input"
    )

    # Update session state if the number of trials changes
    if num_trials != st.session_state.ch_num_trials:
        st.session_state.ch_num_trials = num_trials
        # Reinitialize inputs if num_trials changes significantly
        st.session_state.constant_head_inputs = [
            {"length": 0.0, "area": 0.0, "head": 0.0, "volume": 0.0, "time": 0.0, "temperature": 27.0}
            for _ in range(num_trials)
        ]

    # Initialize session state for storing inputs if not already set
    if "constant_head_inputs" not in st.session_state:
        st.session_state.constant_head_inputs = [
            {"length": 0.0, "area": 0.0, "head": 0.0, "volume": 0.0, "time": 0.0, "temperature": 27.0}
            for _ in range(num_trials)
        ]
    # Ensure the list size matches the current num_trials
    while len(st.session_state.constant_head_inputs) < num_trials:
        st.session_state.constant_head_inputs.append(
            {"length": 0.0, "area": 0.0, "head": 0.0, "volume": 0.0, "time": 0.0, "temperature": 27.0}
        )
    st.session_state.constant_head_inputs = st.session_state.constant_head_inputs[:num_trials]


    # Collect inputs for each trial using session state
    for i in range(num_trials):
        st.markdown(f"### Trial {i + 1}")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.session_state.constant_head_inputs[i]["length"] = st.number_input(
                f"Length of specimen L (cm) - Trial {i+1}",
                key=f"ch_length{i}",
                value=st.session_state.constant_head_inputs[i]["length"]
            )
            st.session_state.constant_head_inputs[i]["area"] = st.number_input(
                f"Cross-sectional area A (cm²) - Trial {i+1}",
                key=f"ch_area{i}",
                value=st.session_state.constant_head_inputs[i]["area"]
            )
        with col2:
            st.session_state.constant_head_inputs[i]["head"] = st.number_input(
                f"Constant head h (cm) - Trial {i+1}",
                key=f"ch_head{i}",
                value=st.session_state.constant_head_inputs[i]["head"]
            )
            st.session_state.constant_head_inputs[i]["volume"] = st.number_input(
                f"Volume of water collected Q (cm³) - Trial {i+1}",
                key=f"ch_volume{i}",
                value=st.session_state.constant_head_inputs[i]["volume"]
            )
        with col3:
            st.session_state.constant_head_inputs[i]["time"] = st.number_input(
                f"Time t (s) - Trial {i+1}",
                key=f"ch_time{i}",
                value=st.session_state.constant_head_inputs[i]["time"]
            )
            st.session_state.constant_head_inputs[i]["temperature"] = st.number_input(
                f"Water temperature (°C) - Trial {i+1}",
                key=f"ch_temp{i}",
                value=st.session_state.constant_head_inputs[i]["temperature"]
            )

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_ch_inputs_button"):
        input_df = pd.DataFrame(st.session_state.constant_head_inputs)
        buffer = StringIO()
        input_df.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="constant_head_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Results Button ---
    if st.button("Calculate Results", key="calculate_ch_results_button"):
        calculated_trial_data = []
        for i, trial_input in enumerate(st.session_state.constant_head_inputs):
            length = trial_input["length"]
            area = trial_input["area"]
            head = trial_input["head"]
            volume = trial_input["volume"]
            time = trial_input["time"]
            temperature = trial_input["temperature"] # Not used in k calculation, but good to store

            if length > 0 and area > 0 and head > 0 and volume > 0 and time > 0:
                k = (volume * length) / (area * head * time)  # in cm/s
                k_mps = k / 100  # convert to m/s

                # Soil type classification based on permeability (in cm/s)
                if k > 1e-1:
                    soil_type = "Gravel"
                elif 1e-2 < k <= 1e-1:
                    soil_type = "Coarse Sand"
                elif 1e-3 < k <= 1e-2:
                    soil_type = "Medium Sand"
                elif 1e-4 < k <= 1e-3:
                    soil_type = "Fine Sand"
                elif 1e-6 < k <= 1e-4:
                    soil_type = "Silt"
                else:
                    soil_type = "Clay"

                calculated_trial_data.append({
                    "Trial": i + 1,
                    "Length (cm)": length,
                    "Area (cm²)": area,
                    "Head (cm)": head,
                    "Volume (cm³)": volume,
                    "Time (s)": time,
                    "k (cm/s)": round(k, 5),
                    "k (m/s)": f"{k_mps:.2e}",
                    "Soil Type": soil_type
                })
            else:
                st.warning(f"Trial {i+1}: Please ensure all input values (Length, Area, Head, Volume, Time) are greater than zero to calculate permeability.")

        if calculated_trial_data:
            results_df = pd.DataFrame(calculated_trial_data)
            st.markdown("### 📋 Results Table")
            st.dataframe(results_df, use_container_width=True)

            avg_k_cm_s = results_df["k (cm/s)"].mean()
            avg_k_m_s = avg_k_cm_s / 100

            st.success(f"**Average Coefficient of Permeability (k)**: {avg_k_cm_s:.5f} cm/s or {avg_k_m_s:.2e} m/s")

            st.markdown("### 🧪 Interpretation Table (Typical k values)")
            interp_df = pd.DataFrame({
                "Soil Type": ["Gravel", "Sand (coarse)", "Sand (medium)", "Sand (fine)", "Silt", "Clay"],
                "Typical k (cm/s)": [">10⁻¹", "10⁻² to 10⁻¹", "10⁻³ to 10⁻²", "10⁻⁴ to 10⁻³", "10⁻⁵ to 10⁻⁶", "<10⁻⁷"]
            })
            st.table(interp_df)

            # Return results for the main app to collect
            return {
                "Test Data": results_df,
                "Average k (cm/s)": f"{avg_k_cm_s:.5f}",
                "Average k (m/s)": f"{avg_k_m_s:.2e}",
                "Interpretation Table": interp_df,
                "Remarks": "Coefficient of permeability determined using the Constant Head method, suitable for coarse-grained soils."
            }
        else:
            st.error("No valid trial data to calculate results. Please check your inputs.")
            return None # Return None if no valid data
    
    return None # Return None initially or if calculation button not pressed