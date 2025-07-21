import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO # Import StringIO for CSV buffer

def run():
    st.subheader("Liquid Limit - Casagrande Method (IS 2720 Part 5)")
    st.markdown("""
    - Determine the water content at which soil starts to flow under a standard impact (25 blows).
    - Plot flow curve (semi-log graph) to determine Liquid Limit at 25 blows.
    """)

    num_samples = st.number_input("Number of Samples", min_value=3, max_value=10, value=4)
    data = {
        "Trial": [f"Trial {i+1}" for i in range(num_samples)],
        "Number of Blows": [0]*num_samples,
        "Moisture Content (%)": [0.0]*num_samples
    }
    
    # Initialize session state for the DataFrame to retain edited data
    if "casagrande_df" not in st.session_state or len(st.session_state.casagrande_df) != num_samples:
        st.session_state.casagrande_df = pd.DataFrame(data)

    edited_df = st.data_editor(st.session_state.casagrande_df, num_rows="dynamic", use_container_width=True)
    st.session_state.casagrande_df = edited_df # Update session state with the edited DataFrame

    # Add the "Save Inputs" button logic here
    if st.button("💾 Save Inputs", key="save_casagrande_inputs_button"):
        # Use the current state of the edited_df for saving
        input_df_to_save = st.session_state.casagrande_df.copy()

        buffer = StringIO()
        input_df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="casagrande_inputs.csv",
            mime="text/csv"
        )


    if st.button("Calculate Liquid Limit"):
        df = st.session_state.casagrande_df.copy() # Use the DataFrame from session state
        
        # Filter out rows where both 'Number of Blows' and 'Moisture Content (%)' are zero
        # This prevents issues with log(0) and fitting if empty rows are present
        df_filtered = df[(df["Number of Blows"] != 0) & (df["Moisture Content (%)"] != 0.0)]

        if df_filtered.empty or len(df_filtered) < 2:
            st.error("Please enter at least two valid data points (non-zero blows and moisture content) to perform the calculation.")
            return None

        try:
            df_sorted = df_filtered.sort_values("Number of Blows")
            x = np.log10(df_sorted["Number of Blows"].astype(float))
            y = df_sorted["Moisture Content (%)"].astype(float)

            # Check if there are enough unique x values for polyfit
            if len(np.unique(x)) < 2:
                st.error("Please provide at least two distinct 'Number of Blows' values for a meaningful calculation.")
                return None
            
            coeffs = np.polyfit(x, y, 1)
            a, b = coeffs
            
            # Ensure x_vals cover a reasonable range for plotting, including 25 blows
            min_x_plot = min(x)
            max_x_plot = max(x)
            # Extend plot range slightly if needed to clearly show 25 blows
            plot_range_min = min(min_x_plot, np.log10(15)) # Ensure lower range
            plot_range_max = max(max_x_plot, np.log10(35)) # Ensure upper range
            x_vals = np.linspace(plot_range_min, plot_range_max, 100)
            y_vals = a * x_vals + b

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, 'ro', label='Observed Data')
            ax.plot(x_vals, y_vals, 'b-', label='Flow Curve (Regression)')
            
            # Mark the Liquid Limit at 25 blows
            LL = a * np.log10(25) + b
            ax.plot(np.log10(25), LL, 'go', markersize=8, label=f'Liquid Limit ({LL:.2f}%)')
            ax.vlines(np.log10(25), 0, LL, color='gray', linestyle='--', linewidth=0.7)
            ax.hlines(LL, min(ax.get_xlim()), np.log10(25), color='gray', linestyle='--', linewidth=0.7)


            ax.set_xlabel("Number of Blows (Log Scale)")
            ax.set_ylabel("Moisture Content (%)")
            ax.set_title("Casagrande Flow Curve")
            ax.legend()
            ax.grid(True, which="both", ls="--", c='0.7')
            
            # Set custom x-axis ticks to show actual blow counts on log scale
            # Choose appropriate tick values (e.g., 10, 15, 20, 25, 30, 40)
            blow_ticks = [10, 15, 20, 25, 30, 40, 50, 60]
            log_blow_ticks = [np.log10(b) for b in blow_ticks]
            ax.set_xticks(log_blow_ticks)
            ax.set_xticklabels([str(b) for b in blow_ticks])

            st.pyplot(fig)

            st.success(f"Estimated Liquid Limit = {LL:.2f}%")

            img_buf = BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight") # Use bbox_inches="tight" for better saving
            img_buf.seek(0)
            plt.close(fig) # Close the figure to free memory

            result_table = pd.DataFrame({
                "Number of Blows": df_sorted["Number of Blows"],
                "Moisture Content (%)": df_sorted["Moisture Content (%)"]
            }).reset_index(drop=True) # Reset index to avoid issues if rows were filtered

            return {
                "Test Data": result_table,
                "Flow Curve Graph": img_buf, # Renamed key for clarity in report
                "Liquid Limit (LL)": f"{LL:.2f}%", # Added LL as a separate string entry
                "Remarks": "Liquid Limit is determined by finding the moisture content corresponding to 25 blows on the flow curve."
            }

        except ValueError as ve:
            st.error(f"Input Data Error: {ve}. Please ensure 'Number of Blows' are positive and 'Moisture Content (%)' values are valid.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during calculation: {e}")
            st.info("Please check your input values. Ensure you have at least two distinct points with non-zero blows and moisture content.")
            return None

    return None