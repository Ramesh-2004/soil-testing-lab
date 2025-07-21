import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO, StringIO # Import StringIO for CSV buffer

def run():
    st.subheader("Specific Gravity by Density Bottle Method (IS 2720: Part 3, Sec 1: 1980)")

    st.markdown("""
    **This test uses carbon tetrachloride (CCl₄) to determine the specific gravity of soil solids.**
    - W₁ = Weight of empty density bottle (g)
    - W₂ = Weight of bottle + dry soil (g)
    - W₃ = Weight of bottle + soil + CCl₄ (g)
    - W₄ = Weight of bottle + CCl₄ only (g)
    - V  = Volume of density bottle (cm³)
    """)

    # --- Session State Initialization for Inputs ---
    if "sg_volume" not in st.session_state:
        st.session_state.sg_volume = 50.0
    if "sg_num_trials" not in st.session_state:
        st.session_state.sg_num_trials = 3

    # Volume of Density Bottle
    st.session_state.sg_volume = st.number_input(
        "Volume of Density Bottle (cm³)",
        min_value=1.0,
        value=st.session_state.sg_volume,
        key="sg_volume_input"
    )

    # Number of Trials
    num_trials = st.number_input(
        "Number of Trials",
        min_value=1, max_value=10,
        value=st.session_state.sg_num_trials,
        step=1,
        key="sg_num_trials_input"
    )

    # Update session state if num_trials changes and reinitialize trial inputs
    if num_trials != st.session_state.sg_num_trials:
        st.session_state.sg_num_trials = num_trials
        # Reinitialize sg_trial_inputs if num_trials changes
        st.session_state.sg_trial_inputs = [
            {"W1": 0.0, "W2": 0.0, "W3": 0.0, "W4": 0.0}
            for _ in range(num_trials)
        ]

    # Initialize trial inputs in session state
    if "sg_trial_inputs" not in st.session_state:
        st.session_state.sg_trial_inputs = [
            {"W1": 0.0, "W2": 0.0, "W3": 0.0, "W4": 0.0}
            for _ in range(num_trials)
        ]
    # Ensure sg_trial_inputs list size matches current num_trials
    while len(st.session_state.sg_trial_inputs) < num_trials:
        st.session_state.sg_trial_inputs.append({"W1": 0.0, "W2": 0.0, "W3": 0.0, "W4": 0.0})
    st.session_state.sg_trial_inputs = st.session_state.sg_trial_inputs[:num_trials]


    st.markdown("### Enter Data for Each Trial")
    for i in range(num_trials):
        st.markdown(f"#### Trial {i+1}")
        col1, col2 = st.columns(2) # Using 2 columns for better layout
        with col1:
            st.session_state.sg_trial_inputs[i]["W1"] = st.number_input(
                f"W₁: Empty Bottle (g) [{i+1}]",
                key=f"sg_W1_{i}",
                value=st.session_state.sg_trial_inputs[i]["W1"]
            )
            st.session_state.sg_trial_inputs[i]["W2"] = st.number_input(
                f"W₂: Bottle + Dry Soil (g) [{i+1}]",
                key=f"sg_W2_{i}",
                value=st.session_state.sg_trial_inputs[i]["W2"]
            )
        with col2:
            st.session_state.sg_trial_inputs[i]["W3"] = st.number_input(
                f"W₃: Bottle + Soil + CCl₄ (g) [{i+1}]",
                key=f"sg_W3_{i}",
                value=st.session_state.sg_trial_inputs[i]["W3"]
            )
            st.session_state.sg_trial_inputs[i]["W4"] = st.number_input(
                f"W₄: Bottle + CCl₄ Only (g) [{i+1}]",
                key=f"sg_W4_{i}",
                value=st.session_state.sg_trial_inputs[i]["W4"]
            )

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_sg_inputs_button"):
        # Create a DataFrame from the current session state inputs
        input_df_to_save = pd.DataFrame(st.session_state.sg_trial_inputs)
        input_df_to_save.insert(0, "Trial", range(1, len(input_df_to_save) + 1)) # Add Trial column

        # Add bottle volume as a separate row or column if desired, for simplicity here, it's implied in context
        # For a more structured save, you might save mould dimensions separately or add them as columns to each row.
        # For now, let's keep it simple and just save the trial data.

        buffer = StringIO()
        input_df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="specific_gravity_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Specific Gravity Button ---
    if st.button("Calculate Specific Gravity", key="calculate_sg_button"):
        results = []
        valid_G_list = []
        
        volume = st.session_state.sg_volume # Use the volume from session state

        if volume <= 0:
            st.error("Volume of Density Bottle must be greater than zero.")
            return None

        for i, trial_input in enumerate(st.session_state.sg_trial_inputs):
            W1 = trial_input["W1"]
            W2 = trial_input["W2"]
            W3 = trial_input["W3"]
            W4 = trial_input["W4"]

            # Input validation for each trial
            if not (W1 >= 0 and W2 >= 0 and W3 >= 0 and W4 >= 0 and W2 > W1 and W4 > W1 and W3 >= W2):
                st.warning(f"Trial {i+1}: Incomplete or invalid input data. Please check W1, W2, W3, W4 values and their relationships.")
                results.append({"Trial": i+1, "Gc": None, "G": None, "Error": "Invalid inputs"})
                continue # Skip calculation for this trial

            try:
                Gc = (W4 - W1) / volume
                denominator = (W4 - W1) - (W3 - W2)

                if denominator != 0:
                    G = (W2 - W1) * Gc / denominator
                    valid_G_list.append(G)
                    results.append({"Trial": i+1, "Gc": round(Gc, 3), "G": round(G, 3), "Error": None})
                else:
                    st.warning(f"Trial {i+1}: Denominator for Specific Gravity calculation is zero. Check inputs (W4-W1) - (W3-W2).")
                    results.append({"Trial": i+1, "Gc": round(Gc, 3), "G": None, "Error": "Denominator is zero"})
            except Exception as e:
                st.error(f"Trial {i+1}: An error occurred during calculation: {e}. Please check inputs.")
                results.append({"Trial": i+1, "Gc": None, "G": None, "Error": str(e)})


        if valid_G_list:
            G_avg = sum(valid_G_list) / len(valid_G_list)

            st.markdown("## 🔍 Results per Trial")
            results_df = pd.DataFrame(results)
            # Filter out the 'Error' column for display if it's all None
            if 'Error' in results_df.columns and results_df['Error'].isnull().all():
                results_df = results_df.drop(columns=['Error'])
            st.dataframe(results_df.round(3), use_container_width=True)


            st.success(f"**Average Specific Gravity (Gₐᵥg) = {G_avg:.3f}**")

            # Determine Soil Type from Table
            soil_type_classification = ""
            st.markdown("### 🧾 Soil Type Interpretation (Based on Gₐᵥg)")
            if G_avg < 2.60:
                soil_type_classification = "Likely contains organic matter."
                st.warning("Soil Type: " + soil_type_classification)
            elif 2.60 <= G_avg <= 2.67:
                soil_type_classification = "Sand or most inorganic soils."
                st.success("Soil Type: " + soil_type_classification)
            elif 2.67 < G_avg <= 2.78:
                soil_type_classification = "Silty sand or clay."
                st.info("Soil Type: " + soil_type_classification)
            elif G_avg > 2.78:
                soil_type_classification = "Possibly dense clay or heavy minerals."
                st.info("Soil Type: " + soil_type_classification)
            else:
                soil_type_classification = "Could not determine soil type."
                st.error(soil_type_classification)

            # Reference Table
            st.markdown("### 📊 Reference Table (IS 2720)")
            ref_df = pd.DataFrame({
                "Soil Type": [
                    "Most Inorganic Soils",
                    "Sand",
                    "Silty Sand",
                    "Clay",
                    "Soils with Organic Matter"
                ],
                "Specific Gravity Range": [
                    "2.60 – 2.80",
                    "2.65 – 2.67",
                    "2.67 – 2.78",
                    "2.70 – 2.80",
                    "< 2.60"
                ]
            })
            st.table(ref_df)

            # Return results for the main app to collect
            return {
                "Bottle Volume (cm³)": st.session_state.sg_volume,
                "Raw Input Data": pd.DataFrame(st.session_state.sg_trial_inputs).insert(0, "Trial", range(1, len(st.session_state.sg_trial_inputs) + 1)),
                "Calculated Trial Results": results_df,
                "Average Specific Gravity (G)": f"{G_avg:.3f}",
                "Soil Type Classification": soil_type_classification,
                "Reference Table": ref_df,
                "Remarks": "Specific Gravity determined using the Density Bottle Method with Carbon Tetrachloride."
            }
        else:
            st.error("No valid specific gravity values could be calculated from the provided trials. Please check your inputs.")
            return None
    
    return None # Default return if calculation button is not pressed