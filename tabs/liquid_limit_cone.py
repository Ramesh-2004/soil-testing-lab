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

    # Update session state if num_trials changes and reinitialize trial inputs
    if num_trials != st.session_state.cone_num_trials:
        st.session_state.cone_num_trials = num_trials
        # Reinitialize cone_trial_inputs if num_trials changes
        st.session_state.cone_trial_inputs = [
            {"penetration": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0}
            for _ in range(num_trials)
        ]
        # st.experimental_rerun() # No need for rerun here, value update is enough

    # Initialize trial inputs in session state
    if "cone_trial_inputs" not in st.session_state:
        st.session_state.cone_trial_inputs = [
            {"penetration": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0}
            for _ in range(num_trials)
        ]
    # Ensure cone_trial_inputs list size matches current num_trials
    while len(st.session_state.cone_trial_inputs) < num_trials:
        st.session_state.cone_trial_inputs.append({"penetration": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0})
    st.session_state.cone_trial_inputs = st.session_state.cone_trial_inputs[:num_trials]


    # --- Input Fields (outside of form for "Save Inputs" button to work independently) ---
    for i in range(num_trials):
        st.markdown(f"### Trial {i+1}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.session_state.cone_trial_inputs[i]["penetration"] = st.number_input(
                f"Penetration (mm) [{i+1}]",
                min_value=0.0,
                key=f"cone_p_{i}",
                value=st.session_state.cone_trial_inputs[i]["penetration"]
            )
        with col2:
            st.session_state.cone_trial_inputs[i]["w1"] = st.number_input(
                f"w1 (Empty Container) [{i+1}]",
                min_value=0.0,
                key=f"cone_w1_{i}",
                value=st.session_state.cone_trial_inputs[i]["w1"]
            )
        with col3:
            st.session_state.cone_trial_inputs[i]["w2"] = st.number_input(
                f"w2 (Wet + Container) [{i+1}]",
                min_value=0.0,
                key=f"cone_w2_{i}",
                value=st.session_state.cone_trial_inputs[i]["w2"]
            )
        with col4:
            st.session_state.cone_trial_inputs[i]["w3"] = st.number_input(
                f"w3 (Dry + Container) [{i+1}]",
                min_value=0.0,
                key=f"cone_w3_{i}",
                value=st.session_state.cone_trial_inputs[i]["w3"]
            )

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_cone_inputs_button"):
        input_df_to_save = pd.DataFrame(st.session_state.cone_trial_inputs)
        # Add trial numbers for clarity in the saved CSV
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

    # --- Calculate Button (inside a form or just a regular button) ---
    # Using a regular button here, as inputs are managed by session_state
    if st.button("Calculate Liquid Limit", key="calculate_cone_ll_button"):
        # Create DataFrame from current session state inputs
        df = pd.DataFrame(st.session_state.cone_trial_inputs)

        # Validate inputs before calculation
        valid_data_points = []
        for i, row in df.iterrows():
            p, w1, w2, w3 = row["penetration"], row["w1"], row["w2"], row["w3"]
            if p > 0 and w1 >= 0 and w2 > w1 and w3 > w1: # w1 can be 0 if it's just the container without soil
                try:
                    weight_of_dry_soil = w3 - w1
                    if weight_of_dry_soil > 0:
                        water_content = ((w2 - w3) / weight_of_dry_soil) * 100
                        valid_data_points.append({"Penetration (mm)": p, "Water Content (%)": water_content})
                    else:
                        st.warning(f"Trial {i+1}: Weight of dry soil (W3-W1) is zero or negative. Cannot calculate water content.")
                except Exception as e:
                    st.warning(f"Trial {i+1}: Error calculating water content: {e}. Check inputs.")
            else:
                st.warning(f"Trial {i+1}: Incomplete or invalid input data. Check penetration, w1, w2, w3 values.")

        if not valid_data_points or len(valid_data_points) < 2:
            st.error("Please enter at least two valid data points with non-zero penetration and moisture content for calculation.")
            return None

        df_calculated = pd.DataFrame(valid_data_points)
        
        try:
            # Fit a polynomial (typically 2nd degree for better fit, but 1st is also common)
            # Ensure x and y values are floats
            x_data = df_calculated["Penetration (mm)"].astype(float)
            y_data = df_calculated["Water Content (%)"].astype(float)

            # Check for sufficient unique points for polynomial fitting
            if len(np.unique(x_data)) < 2:
                st.error("Not enough unique penetration values to fit a curve. Please provide at least two distinct penetration readings.")
                return None
            # If all y are same, but x are different, polyfit might still work for degree 1
            if len(np.unique(y_data)) < 2 and len(np.unique(x_data)) > 1:
                st.warning("All water content values are the same. A meaningful curve fit may not be possible.")


            coeffs = np.polyfit(x_data, y_data, 1) # Using 1st degree for simplicity, as per Casagrande method's linear fit on log scale.
                                                    # If a curve is consistently observed, 2 might be better.
            poly = np.poly1d(coeffs)
            liquid_limit = poly(20) # Liquid Limit is defined at 20mm penetration for cone method

            st.write("### Trial Data and Calculated Water Content")
            # Display the DataFrame with calculated water content
            display_df = df_calculated.copy()
            # Merge original inputs for full display if needed, or just show calculated
            original_inputs_df = pd.DataFrame(st.session_state.cone_trial_inputs)
            # Add Trial column to original_inputs_df for the report
            original_inputs_df.insert(0, "Trial", range(1, len(original_inputs_df) + 1))

            display_df = pd.merge(original_inputs_df, display_df, on=["Penetration (mm)"], how="left", suffixes=('_input', ''))
            st.dataframe(display_df.round(2), use_container_width=True)

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
                "Input Data": original_inputs_df, # Raw inputs with Trial numbers
                "Calculated Data": df_calculated, # Penetration vs Water Content
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
            return None

    return None # Default return if calculation button is not pressed