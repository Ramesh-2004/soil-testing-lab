import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO # Import StringIO for CSV buffer

def run():
    st.subheader("Liquid Limit by Cone Penetration Method (IS 2720: Part 5: 1985)")

    st.markdown("""
    For each trial, enter the following:
    - Penetration (mm)
    - Weight of empty container (w1)
    - Weight of container + wet soil (w2)
    - Weight of container + dry soil (w3)
    """)

    # --- Helper function to calculate Moisture Content ---
    def calculate_moisture_content(w1, w2, w3):
        """
        Calculates moisture content (in %) from weights.
        w1: Weight of empty container (g)
        w2: Weight of container + wet soil (g)
        w3: Weight of container + dry soil (g)
        """
        # If all inputs are zero, assume no data yet and return 0.0
        if w1 == 0.0 and w2 == 0.0 and w3 == 0.0:
            return 0.0
        
        # Basic validation for weights
        # Ensure w2 (wet + container) is greater than w3 (dry + container)
        # And w3 (dry + container) is greater than w1 (empty container)
        if not (w2 > w3 and w3 > w1):
            return np.nan # Return NaN for invalid weight combinations

        weight_of_water = w2 - w3
        weight_of_dry_soil = w3 - w1
        
        if weight_of_dry_soil <= 0: # Avoid division by zero or negative dry soil weight
            return np.nan # Cannot calculate if dry soil weight is zero or negative
        
        return (weight_of_water / weight_of_dry_soil) * 100

    # --- Session State Initialization for Inputs ---
    # Initialize number of trials
    if "cone_num_trials" not in st.session_state:
        st.session_state.cone_num_trials = 4

    num_trials = st.number_input(
        "Number of Trials",
        min_value=3, max_value=10,
        value=st.session_state.cone_num_trials,
        key="cone_trial_input"
    )

    # Initialize trial inputs in session state
    if "cone_trial_inputs" not in st.session_state:
        st.session_state.cone_trial_inputs = [
            {"penetration": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0, "water_content": 0.0} # Added water_content
            for _ in range(num_trials)
        ]
    # Adjust list size if num_trials changes
    if len(st.session_state.cone_trial_inputs) != num_trials:
        # Create a new list, preserving existing data if possible
        new_inputs = []
        for i in range(num_trials):
            if i < len(st.session_state.cone_trial_inputs):
                new_inputs.append(st.session_state.cone_trial_inputs[i])
            else:
                new_inputs.append({"penetration": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0, "water_content": 0.0})
        st.session_state.cone_trial_inputs = new_inputs
        st.session_state.cone_num_trials = num_trials # Update stored num_trials

    # --- Input Fields ---
    for i in range(num_trials):
        st.markdown(f"### Trial {i+1}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.session_state.cone_trial_inputs[i]["penetration"] = st.number_input(
                f"Penetration (mm) [{i+1}]",
                min_value=0.0,
                key=f"cone_p_{i}",
                value=st.session_state.cone_trial_inputs[i]["penetration"],
                format="%.2f"
            )
        with col2:
            st.session_state.cone_trial_inputs[i]["w1"] = st.number_input(
                f"w1 (Empty Container) [{i+1}]",
                min_value=0.0,
                key=f"cone_w1_{i}",
                value=st.session_state.cone_trial_inputs[i]["w1"],
                format="%.2f"
            )
        with col3:
            st.session_state.cone_trial_inputs[i]["w2"] = st.number_input(
                f"w2 (Wet + Container) [{i+1}]",
                min_value=0.0,
                key=f"cone_w2_{i}",
                value=st.session_state.cone_trial_inputs[i]["w2"],
                format="%.2f"
            )
        with col4:
            st.session_state.cone_trial_inputs[i]["w3"] = st.number_input(
                f"w3 (Dry + Container) [{i+1}]",
                min_value=0.0,
                key=f"cone_w3_{i}",
                value=st.session_state.cone_trial_inputs[i]["w3"],
                format="%.2f"
            )
        
        # Calculate water content immediately after inputs are updated
        w1_val = st.session_state.cone_trial_inputs[i]["w1"]
        w2_val = st.session_state.cone_trial_inputs[i]["w2"]
        w3_val = st.session_state.cone_trial_inputs[i]["w3"]
        calculated_wc = calculate_moisture_content(w1_val, w2_val, w3_val)
        st.session_state.cone_trial_inputs[i]["water_content"] = calculated_wc

        if np.isnan(calculated_wc):
            st.error(f"Trial {i+1} Water Content: Invalid input for calculation.")
        else:
            st.info(f"Trial {i+1} Water Content: **{calculated_wc:.2f}%**")


    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_cone_inputs_button"):
        input_df_to_save = pd.DataFrame(st.session_state.cone_trial_inputs)
        input_df_to_save.insert(0, "Trial", range(1, len(input_df_to_save) + 1))

        buffer = StringIO()
        input_df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="cone_penetration_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Button ---
    if st.button("Calculate Liquid Limit", key="calculate_cone_ll_button"):
        # Create DataFrame from current session state inputs (including calculated water_content)
        df_all_trials = pd.DataFrame(st.session_state.cone_trial_inputs)
        df_all_trials.insert(0, "Trial", range(1, len(df_all_trials) + 1)) # Add Trial column

        # Filter for valid data points for curve fitting
        # A point is valid if penetration > 0 and water_content is a valid number (>0 or not NaN)
        df_calculated_for_fit = df_all_trials[
            (df_all_trials["penetration"] > 0) &
            (~df_all_trials["water_content"].isna()) &
            (df_all_trials["water_content"] > 0) # Ensure water content is positive
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if df_calculated_for_fit.empty or len(df_calculated_for_fit) < 2:
            st.error("Please enter at least two valid data points with non-zero penetration and calculable, positive water content for calculation.")
            return None

        try:
            x_data = df_calculated_for_fit["penetration"].astype(float)
            y_data = df_calculated_for_fit["water_content"].astype(float)

            if len(np.unique(x_data)) < 2:
                st.error("Not enough unique penetration values to fit a curve. Please provide at least two distinct penetration readings.")
                return None
            
            # If all y are same, but x are different, polyfit might still work for degree 1
            if len(np.unique(y_data)) < 2 and len(np.unique(x_data)) > 1:
                st.warning("All water content values are the same. A meaningful curve fit may not be possible.")

            coeffs = np.polyfit(x_data, y_data, 1) # 1st degree polynomial (linear fit)
            poly = np.poly1d(coeffs)
            liquid_limit = poly(20) # Liquid Limit is defined at 20mm penetration for cone method

            st.write("### Trial Data and Calculated Water Content")
            # Display the full DataFrame including original inputs and calculated water content
            st.dataframe(df_all_trials.round(2), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x_data, y_data, color='blue', label='Observed Data')
            
            # Plot fitted curve over a reasonable range
            x_vals_plot = np.linspace(min(x_data) - 5, max(x_data) + 5, 100) # Extend range slightly
            y_vals_plot = poly(x_vals_plot)
            ax.plot(x_vals_plot, y_vals_plot, color='green', linestyle='--', label='Fitted Curve')
            
            # Mark the Liquid Limit at 20 mm Penetration
            ax.axvline(20, color='red', linestyle=':', label='20 mm Penetration')
            ax.axhline(liquid_limit, color='orange', linestyle=':', label=f'LL = {liquid_limit:.2f}%')
            ax.plot(20, liquid_limit, 'ro', markersize=8, label='Liquid Limit Point') # Mark the LL point

            ax.set_xlabel("Penetration (mm)")
            ax.set_ylabel("Water Content (%)")
            ax.set_title("Cone Penetration Method: Flow Curve")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            img_buf = BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            img_buf.seek(0)
            plt.close(fig) # Close the figure to free memory

            st.markdown("### Result")
            st.success(f"Liquid Limit (Cone Penetration Method) = {liquid_limit:.2f}%")

            # Soil Classification
            soil_classification = ""
            st.markdown("### Soil Classification Based on Liquid Limit")
            if liquid_limit < 35:
                soil_classification = "Low Plasticity Soil"
                st.info(soil_classification)
            elif 35 <= liquid_limit <= 50:
                soil_classification = "Intermediate Plasticity Soil"
                st.warning(soil_classification)
            else:
                soil_classification = "High Plasticity Soil"
                st.success(soil_classification)

            # Return results for the main app to collect
            return {
                "Input Data": df_all_trials, # Raw inputs with Trial numbers and calculated WC
                "Calculated Data Used for Fit": df_calculated_for_fit, # Only valid points
                "Flow Curve Graph": img_buf,
                "Liquid Limit (LL)": f"{liquid_limit:.2f}%",
                "Soil Classification": soil_classification,
                "Remarks": "Liquid Limit determined using the Cone Penetration Method at 20mm penetration."
            }

        except np.linalg.LinAlgError:
            st.error("Cannot fit a curve. This usually happens if all penetration values are the same, or if there's not enough variation in data.")
            st.info("Please ensure you have at least two distinct penetration values and corresponding water contents.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during curve fitting or calculation: {e}")
            st.info("Please check your input values. Ensure w2 > w3 > w1 for valid water content calculation.")
            return None

    return None # Default return if calculation button is not pressed