import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO

def run():
    st.subheader("Liquid Limit - Casagrande Method (IS 2720 Part 5)")
    st.markdown("""
    - Determine the water content at which soil starts to flow under a standard impact (25 blows).
    - Plot flow curve (semi-log graph) to determine Liquid Limit at 25 blows.
    """)

    # --- Helper function to calculate Moisture Content ---
    def calculate_moisture_content(w1, w2, w3):
        """
        Calculates moisture content (in %) from weights.
        w1: Weight of empty cup (g)
        w2: Weight of cup + wet soil (g)
        w3: Weight of cup + dry soil (g)
        """
        # Return 0.0 or NaN if all inputs are zero, indicating no data entered yet
        if w1 == 0.0 and w2 == 0.0 and w3 == 0.0:
            return 0.0 # Display 0.0 initially for empty rows
        
        # Basic validation for weights
        # Ensure wet soil + cup >= dry soil + cup >= empty cup
        if not (w2 >= w3 and w3 >= w1):
            return np.nan # Return NaN for invalid weight combinations (e.g., wet < dry)
        
        weight_of_water = w2 - w3
        weight_of_dry_soil = w3 - w1
        
        if weight_of_dry_soil <= 0: # Avoid division by zero or negative dry soil weight
            return np.nan # Cannot calculate if dry soil weight is zero or negative
        
        return (weight_of_water / weight_of_dry_soil) * 100

    # --- Input section for Number of Samples ---
    num_samples = st.number_input(
        "Number of Trials/Samples",
        min_value=2, # Minimum 2 samples needed for regression
        max_value=10,
        value=4,
        key="num_samples_input"
    )

    st.markdown("---")
    st.write("### Enter Data for Each Trial:")

    # Initialize session state for individual inputs
    if "trial_data" not in st.session_state:
        st.session_state.trial_data = {}
        # Pre-populate some initial structure for the default number of samples
        for i in range(num_samples):
            st.session_state.trial_data[f"trial_{i+1}"] = {
                "Number of Blows": 0.0,
                "Weight of empty cup (g)": 0.0,
                "Weight of cup + wet soil (g)": 0.0,
                "Weight of cup + dry soil (g)": 0.0,
                "Moisture Content (%)": 0.0 # Will be calculated
            }
    
    # Adjust session state if num_samples changes (add or remove trials)
    current_trial_keys = list(st.session_state.trial_data.keys())
    # Add new trials if num_samples increased
    for i in range(len(current_trial_keys), num_samples):
        st.session_state.trial_data[f"trial_{i+1}"] = {
            "Number of Blows": 0.0,
            "Weight of empty cup (g)": 0.0,
            "Weight of cup + wet soil (g)": 0.0,
            "Weight of cup + dry soil (g)": 0.0,
            "Moisture Content (%)": 0.0
        }
    # Remove trials if num_samples decreased
    if len(current_trial_keys) > num_samples:
        keys_to_remove = [f"trial_{i+1}" for i in range(num_samples, len(current_trial_keys))]
        for key in keys_to_remove:
            if key in st.session_state.trial_data:
                del st.session_state.trial_data[key]


    # Create input fields for each trial
    for i in range(num_samples):
        trial_key = f"trial_{i+1}"
        
        # Only ONE st.expander call per trial
        with st.expander(f"Trial {i+1} Details", expanded=True): # Ensure it's expanded by default
            # Apply styling once at the beginning if desired, or keep it in the loop if you want per-expander control
            if i == 0: # Apply styling only once, or adjust as needed
                 st.markdown(f"""
                    <style>
                        div[data-testid="stExpander"] div[role="button"] p {{
                            font-weight: bold;
                            font-size: 1.1em;
                        }}
                    </style>
                """, unsafe_allow_html=True)

            st.session_state.trial_data[trial_key]["Number of Blows"] = st.number_input(
                f"Number of Blows (Trial {i+1})",
                min_value=0.0,
                value=float(st.session_state.trial_data[trial_key]["Number of Blows"]),
                key=f"blows_{i+1}",
                format="%.1f"
            )
            st.session_state.trial_data[trial_key]["Weight of empty cup (g)"] = st.number_input(
                f"Weight of empty cup (g) (Trial {i+1})",
                min_value=0.0,
                value=float(st.session_state.trial_data[trial_key]["Weight of empty cup (g)"]),
                key=f"w1_{i+1}",
                format="%.2f"
            )
            st.session_state.trial_data[trial_key]["Weight of cup + wet soil (g)"] = st.number_input(
                f"Weight of cup + wet soil (g) (Trial {i+1})",
                min_value=0.0,
                value=float(st.session_state.trial_data[trial_key]["Weight of cup + wet soil (g)"]),
                key=f"w2_{i+1}",
                format="%.2f"
            )
            st.session_state.trial_data[trial_key]["Weight of cup + dry soil (g)"] = st.number_input(
                f"Weight of cup + dry soil (g) (Trial {i+1})",
                min_value=0.0,
                value=float(st.session_state.trial_data[trial_key]["Weight of cup + dry soil (g)"]),
                key=f"w3_{i+1}",
                format="%.2f"
            )
            
            # Calculate and display moisture content for current trial
            w1_val = st.session_state.trial_data[trial_key]["Weight of empty cup (g)"]
            w2_val = st.session_state.trial_data[trial_key]["Weight of cup + wet soil (g)"]
            w3_val = st.session_state.trial_data[trial_key]["Weight of cup + dry soil (g)"]
            
            calculated_mc = calculate_moisture_content(w1_val, w2_val, w3_val)
            st.session_state.trial_data[trial_key]["Moisture Content (%)"] = calculated_mc

            if np.isnan(calculated_mc):
                st.error(f"Moisture Content (Trial {i+1}): Invalid Input for Calculation")
            else:
                st.info(f"Moisture Content (Trial {i+1}): **{calculated_mc:.2f}%**")

    st.markdown("---")

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_inputs_button"):
        # Convert the session_state.trial_data dictionary to a DataFrame for saving
        df_to_save = pd.DataFrame.from_dict(st.session_state.trial_data, orient='index')
        df_to_save.index.name = "Trial Key"
        df_to_save = df_to_save.reset_index().rename(columns={"index": "Trial"})

        buffer = StringIO()
        df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="casagrande_inputs_individual.csv",
            mime="text/csv"
        )

    # --- Calculate Liquid Limit Button ---
    if st.button("Calculate Liquid Limit"):
        # Convert collected individual trial data into a DataFrame for processing
        df = pd.DataFrame.from_dict(st.session_state.trial_data, orient='index')
        df = df.reset_index().rename(columns={"index": "Trial"}) # Rename index to 'Trial' column

        # Filter out rows where 'Number of Blows' is zero or 'Moisture Content (%)' is invalid/NaN/zero
        df_filtered = df[(df["Number of Blows"] != 0.0) & (~df["Moisture Content (%)"].isna()) & (df["Moisture Content (%)"] != 0.0)]

        if df_filtered.empty or len(df_filtered) < 2:
            st.error("Please enter at least two valid data points (non-zero blows and calculable moisture content) to perform the calculation.")
            return None

        try:
            df_sorted = df_filtered.sort_values("Number of Blows")
            x = np.log10(df_sorted["Number of Blows"].astype(float))
            y = df_sorted["Moisture Content (%)"].astype(float)

            if len(np.unique(x)) < 2:
                st.error("Please provide at least two distinct 'Number of Blows' values for a meaningful calculation.")
                return None
            
            coeffs = np.polyfit(x, y, 1)
            a, b = coeffs
            
            min_x_plot = min(x)
            max_x_plot = max(x)
            plot_range_min = min(min_x_plot, np.log10(15))
            plot_range_max = max(max_x_plot, np.log10(35))
            x_vals = np.linspace(plot_range_min, plot_range_max, 100)
            y_vals = a * x_vals + b

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, 'ro', label='Observed Data')
            ax.plot(x_vals, y_vals, 'b-', label='Flow Curve (Regression)')
            
            LL = a * np.log10(25) + b
            ax.plot(np.log10(25), LL, 'go', markersize=8, label=f'Liquid Limit ({LL:.2f}%)')
            ax.vlines(np.log10(25), 0, LL, color='gray', linestyle='--', linewidth=0.7)
            ax.hlines(LL, min(ax.get_xlim()), np.log10(25), color='gray', linestyle='--', linewidth=0.7)

            ax.set_xlabel("Number of Blows (Log Scale)")
            ax.set_ylabel("Moisture Content (%)")
            ax.set_title("Casagrande Flow Curve")
            ax.legend()
            ax.grid(True, which="both", ls="--", c='0.7')
            
            blow_ticks = [10, 15, 20, 25, 30, 40, 50, 60]
            log_blow_ticks = [np.log10(b) for b in blow_ticks]
            ax.set_xticks(log_blow_ticks)
            ax.set_xticklabels([str(b) for b in blow_ticks])

            st.pyplot(fig)

            st.success(f"Estimated Liquid Limit = {LL:.2f}%")

            img_buf = BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            img_buf.seek(0)
            plt.close(fig)

            # Prepare result table using the calculated moisture content
            result_table = df_sorted.copy().reset_index(drop=True)
            
            st.markdown("### Data Used for Calculation:")
            st.dataframe(result_table) # Show the data as a table for review

            return {
                "Test Data": result_table,
                "Flow Curve Graph": img_buf,
                "Liquid Limit (LL)": f"{LL:.2f}%",
                "Remarks": "Liquid Limit is determined by finding the moisture content corresponding to 25 blows on the flow curve."
            }

        except ValueError as ve:
            st.error(f"Input Data Error: {ve}. Please ensure 'Number of Blows' are positive and 'Moisture Content (%)' values are valid.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during calculation: {e}")
            st.info("Please check your input values. Ensure you have at least two distinct points with non-zero blows and calculable moisture content.")
            return None

    return None