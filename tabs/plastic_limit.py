import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO, StringIO # Import StringIO for CSV buffer

def run():
    st.subheader("Plastic Limit (IS 2720: Part 5: 1985)")

    st.markdown("""
    For each trial, enter the following:
    - Weight of Empty Cup (w1)
    - Weight of Cup + Wet Soil (w2)
    - Weight of Cup + Dry Soil (w3)

    Water content will be computed, and Plastic Limit is the average of all trials.
    """)

    # --- Session State Initialization for Inputs ---
    # Initialize number of trials
    if "pl_num_trials" not in st.session_state:
        st.session_state.pl_num_trials = 3

    new_n = st.number_input(
        "Number of Trials",
        min_value=2, max_value=10,
        value=st.session_state.pl_num_trials,
        key="pl_trial_input"
    )

    # Update session state if num_trials changes and reinitialize trial inputs
    if new_n != st.session_state.pl_num_trials:
        st.session_state.pl_num_trials = new_n
        # Reinitialize pl_trial_inputs if num_trials changes
        st.session_state.pl_trial_inputs = [
            {"w1": 0.0, "w2": 0.0, "w3": 0.0}
            for _ in range(new_n)
        ]
    
    # Initialize trial inputs in session state
    if "pl_trial_inputs" not in st.session_state:
        st.session_state.pl_trial_inputs = [
            {"w1": 0.0, "w2": 0.0, "w3": 0.0}
            for _ in range(new_n)
        ]
    # Ensure pl_trial_inputs list size matches current num_trials
    while len(st.session_state.pl_trial_inputs) < new_n:
        st.session_state.pl_trial_inputs.append({"w1": 0.0, "w2": 0.0, "w3": 0.0})
    st.session_state.pl_trial_inputs = st.session_state.pl_trial_inputs[:new_n]


    # --- Input Fields (outside of form for "Save Inputs" button to work independently) ---
    for i in range(st.session_state.pl_num_trials):
        st.markdown(f"**Trial {i+1}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pl_trial_inputs[i]["w1"] = st.number_input(
                f"w1 (Empty Cup) [{i+1}]",
                min_value=0.0,
                key=f"pl_w1_{i}",
                value=st.session_state.pl_trial_inputs[i]["w1"]
            )
        with col2:
            st.session_state.pl_trial_inputs[i]["w2"] = st.number_input(
                f"w2 (Cup + Wet Soil) [{i+1}]",
                min_value=0.0,
                key=f"pl_w2_{i}",
                value=st.session_state.pl_trial_inputs[i]["w2"]
            )
        with col3:
            st.session_state.pl_trial_inputs[i]["w3"] = st.number_input(
                f"w3 (Cup + Dry Soil) [{i+1}]",
                min_value=0.0,
                key=f"pl_w3_{i}",
                value=st.session_state.pl_trial_inputs[i]["w3"]
            )

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_pl_inputs_button"):
        input_df_to_save = pd.DataFrame(st.session_state.pl_trial_inputs)
        # Add trial numbers for clarity in the saved CSV
        input_df_to_save.insert(0, "Trial", range(1, len(input_df_to_save) + 1))

        buffer = StringIO()
        input_df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="plastic_limit_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Button ---
    if st.button("Calculate Plastic Limit", key="calculate_pl_button"):
        calculated_wc = []
        valid_inputs_df_data = [] # To store valid inputs for the report

        for i, trial_input in enumerate(st.session_state.pl_trial_inputs):
            w1 = trial_input["w1"]
            w2 = trial_input["w2"]
            w3 = trial_input["w3"]

            if w1 >= 0 and w2 >= 0 and w3 >= 0 and w2 > w1 and w3 > w1:
                try:
                    weight_of_dry_soil = w3 - w1
                    if weight_of_dry_soil > 0:
                        moisture_content = ((w2 - w3) / weight_of_dry_soil) * 100
                        calculated_wc.append(moisture_content)
                        valid_inputs_df_data.append({
                            "Trial": i + 1,
                            "W1 (Empty Cup)": w1,
                            "W2 (Cup + Wet Soil)": w2,
                            "W3 (Cup + Dry Soil)": w3,
                            "Water Content (%)": round(moisture_content, 2)
                        })
                    else:
                        st.warning(f"Trial {i+1}: Weight of dry soil (W3-W1) is zero or negative. Cannot calculate water content.")
                except Exception as e:
                    st.warning(f"Trial {i+1}: Error calculating water content: {e}. Check inputs.")
            else:
                st.warning(f"Trial {i+1}: Incomplete or invalid input data. Ensure W2 > W1 and W3 > W1.")
        
        if not calculated_wc:
            st.error("No valid water content values could be calculated. Please check your inputs.")
            return None

        # Create DataFrame for display based on valid calculated data
        df_results = pd.DataFrame(valid_inputs_df_data)
        
        avg_pl = np.mean(calculated_wc)

        st.write("### Plastic Limit Trial Data and Calculated Water Content")
        st.dataframe(df_results.style.format({"Water Content (%)": "{:.2f}"}), use_container_width=True)

        st.markdown("### Result")
        st.success(f"Average Plastic Limit = {avg_pl:.2f}%")

        st.markdown("""
        **Note:** Plasticity Index (PI) = LL - PL  
        Please ensure LL is calculated from Casagrande or Cone Penetration method before computing PI.

        Classification as per IS 1498 is based on PI and LL.
        """)

        # Return results for the main app to collect
        return {
            "Input Data": pd.DataFrame(st.session_state.pl_trial_inputs).insert(0, "Trial", range(1, len(st.session_state.pl_trial_inputs) + 1)), # Raw inputs
            "Calculated Water Contents": df_results, # Dataframe with calculated WCs
            "Average Plastic Limit (PL)": f"{avg_pl:.2f}%",
            "Remarks": "Plastic Limit is the average water content at which the soil can be rolled into a 3mm thread without crumbling."
        }
    
    return None # Default return if calculation button is not pressed