import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO # Import StringIO for CSV buffer

def run():
    st.subheader("Variable Head Permeability Test (IS 2720 Part 17:1986)")

    st.markdown(r"""
    This test is used for fine-grained soils. The coefficient of permeability \( k \) is calculated using:

    $$
    k = \frac{2.3 \cdot a \cdot L}{A \cdot t} \cdot \log_{10}\left(\frac{h_1}{h_2}\right)
    $$

    - \( a \): Area of standpipe (cm²)
    - \( A \): Cross-sectional area of soil specimen (cm²)
    - \( L \): Length of specimen (cm)
    - \( t \): Time interval (s)
    - \( h_1, h_2 \): Initial and final heads (cm)
    """)

    # --- Session State Initialization for Inputs ---
    if "vh_num_trials" not in st.session_state:
        st.session_state.vh_num_trials = 3
    if "vh_a" not in st.session_state:
        st.session_state.vh_a = 1.0
    if "vh_A" not in st.session_state:
        st.session_state.vh_A = 50.0
    if "vh_L" not in st.session_state:
        st.session_state.vh_L = 10.0

    # Number of Trials input
    num_trials = st.number_input(
        "Number of Trials",
        min_value=1, max_value=10,
        value=st.session_state.vh_num_trials,
        step=1,
        key="vh_num_trials_input"
    )

    # Update session state for num_trials and reinitialize trial inputs if changed
    if num_trials != st.session_state.vh_num_trials:
        st.session_state.vh_num_trials = num_trials
        # Reinitialize vh_trial_inputs if num_trials changes
        st.session_state.vh_trial_inputs = [
            {"h1": 0.0, "h2": 0.0, "t": 0.0}
            for _ in range(num_trials)
        ]

    # Initialize trial inputs in session state
    if "vh_trial_inputs" not in st.session_state:
        st.session_state.vh_trial_inputs = [
            {"h1": 0.0, "h2": 0.0, "t": 0.0}
            for _ in range(num_trials)
        ]
    # Ensure vh_trial_inputs list size matches current num_trials
    while len(st.session_state.vh_trial_inputs) < num_trials:
        st.session_state.vh_trial_inputs.append({"h1": 0.0, "h2": 0.0, "t": 0.0})
    st.session_state.vh_trial_inputs = st.session_state.vh_trial_inputs[:num_trials]


    # Constant Parameters input, bound to session state
    with st.expander("Enter Constant Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.vh_a = st.number_input(
                "Area of Standpipe (a) in cm²",
                min_value=0.0001, format="%.4f",
                value=st.session_state.vh_a,
                key="vh_a_input"
            )
            st.session_state.vh_A = st.number_input(
                "Cross-sectional Area of Soil Sample (A) in cm²",
                min_value=0.01, format="%.2f",
                value=st.session_state.vh_A,
                key="vh_A_input"
            )
        with col2:
            st.session_state.vh_L = st.number_input(
                "Length of Soil Specimen (L) in cm",
                min_value=0.01, format="%.2f",
                value=st.session_state.vh_L,
                key="vh_L_input"
            )

    # Trial Data Input, bound to session state
    for i in range(num_trials):
        st.markdown(f"### Trial {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.vh_trial_inputs[i]["h1"] = st.number_input(
                f"Initial Head h₁ (cm) - Trial {i+1}",
                min_value=0.0, # Changed to 0.0 to allow for initial empty state, validation will handle later
                value=st.session_state.vh_trial_inputs[i]["h1"],
                key=f"vh_h1_{i}"
            )
        with col2:
            st.session_state.vh_trial_inputs[i]["h2"] = st.number_input(
                f"Final Head h₂ (cm) - Trial {i+1}",
                min_value=0.0, # Changed to 0.0
                value=st.session_state.vh_trial_inputs[i]["h2"],
                key=f"vh_h2_{i}"
            )
        with col3:
            st.session_state.vh_trial_inputs[i]["t"] = st.number_input(
                f"Time Interval (s) - Trial {i+1}",
                min_value=0.0, # Changed to 0.0
                value=st.session_state.vh_trial_inputs[i]["t"],
                key=f"vh_t_{i}"
            )

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_vh_inputs_button"):
        input_data_for_save = []
        # Add constant parameters as first rows or separate section
        input_data_for_save.append({"Parameter": "Area of Standpipe (a) cm²", "Value": st.session_state.vh_a})
        input_data_for_save.append({"Parameter": "Cross-sectional Area of Soil Sample (A) cm²", "Value": st.session_state.vh_A})
        input_data_for_save.append({"Parameter": "Length of Soil Specimen (L) cm", "Value": st.session_state.vh_L})
        input_data_for_save.append({"Parameter": "--- Trial Data ---", "Value": ""}) # Separator

        for i, trial_input in enumerate(st.session_state.vh_trial_inputs):
            input_data_for_save.append({
                "Parameter": f"Trial {i+1} - Initial Head h1 (cm)", "Value": trial_input["h1"]
            })
            input_data_for_save.append({
                "Parameter": f"Trial {i+1} - Final Head h2 (cm)", "Value": trial_input["h2"]
            })
            input_data_for_save.append({
                "Parameter": f"Trial {i+1} - Time Interval (s)", "Value": trial_input["t"]
            })
        
        input_df_to_save = pd.DataFrame(input_data_for_save)
        
        buffer = StringIO()
        input_df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="variable_head_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Results Button ---
    if st.button("Calculate Permeability", key="calculate_vh_results_button"):
        calculated_trials = []
        
        # Get constant parameters from session state
        a = st.session_state.vh_a
        A = st.session_state.vh_A
        L = st.session_state.vh_L

        # Validate constant parameters first
        if not (a > 0 and A > 0 and L > 0):
            st.error("Please ensure 'Area of Standpipe (a)', 'Cross-sectional Area of Soil Sample (A)', and 'Length of Soil Specimen (L)' are all greater than zero.")
            return None

        for i, trial_input in enumerate(st.session_state.vh_trial_inputs):
            h1 = trial_input["h1"]
            h2 = trial_input["h2"]
            t = trial_input["t"]

            # Validate trial-specific inputs
            if not (h1 > 0 and h2 > 0 and t > 0 and h1 > h2):
                st.warning(f"Trial {i+1}: Invalid input. Ensure h₁ > h₂, and all h₁, h₂, t are positive. Skipping this trial.")
                continue # Skip calculation for this trial

            try:
                k = (2.3 * a * L) / (A * t) * math.log10(h1 / h2)
                calculated_trials.append({
                    "Trial": i+1,
                    "h1 (cm)": h1,
                    "h2 (cm)": h2,
                    "Time (s)": t,
                    "k (cm/s)": round(k, 6)
                })
            except Exception as e:
                st.error(f"Trial {i+1}: An error occurred during calculation: {e}. Check inputs.")
                continue # Skip calculation for this trial

        if calculated_trials:
            df_results = pd.DataFrame(calculated_trials)
            st.markdown("### Trial Results")
            st.dataframe(df_results, use_container_width=True)

            avg_k = df_results["k (cm/s)"].mean()
            st.success(f"Average Coefficient of Permeability: {avg_k:.6f} cm/s")

            # Classification
            soil_type_classification = ""
            if avg_k < 1e-7:
                soil_type_classification = "Very low permeability (e.g., Clay)"
            elif avg_k < 1e-5:
                soil_type_classification = "Low permeability (e.g., Silty Clay)"
            elif avg_k < 1e-3:
                soil_type_classification = "Medium permeability (e.g., Silty Sand)"
            else:
                soil_type_classification = "High permeability (e.g., Sand or Gravel)"
            
            st.info(f"**Soil Classification**: {soil_type_classification}")

            # Plot k vs time
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_results["Time (s)"], df_results["k (cm/s)"], marker='o', color='green', linestyle='-')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Permeability (k in cm/s)")
            ax.set_title("Variation of Permeability with Time")
            ax.grid(True)
            st.pyplot(fig)

            img_buf = BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            img_buf.seek(0)
            plt.close(fig) # Close the figure to free memory

            # Return results for the main app to collect
            return {
                "Constant Parameters": pd.DataFrame({
                    "Parameter": ["Area of Standpipe (a) cm²", "Cross-sectional Area (A) cm²", "Length (L) cm"],
                    "Value": [st.session_state.vh_a, st.session_state.vh_A, st.session_state.vh_L]
                }),
                "Trial Input Data": pd.DataFrame(st.session_state.vh_trial_inputs).insert(0, "Trial", range(1, len(st.session_state.vh_trial_inputs) + 1)),
                "Calculated Permeability Data": df_results,
                "Average Permeability (k)": f"{avg_k:.6f} cm/s",
                "Soil Classification": soil_type_classification,
                "Permeability vs Time Graph": img_buf,
                "Remarks": "Coefficient of permeability determined using the Variable Head method, suitable for fine-grained soils."
            }
        else:
            st.error("No valid trials were processed. Please check your inputs and ensure they meet the criteria (h1 > h2, all values positive).")
            return None
    
    return None # Default return if calculation button is not pressed