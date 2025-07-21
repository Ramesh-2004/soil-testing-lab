import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO # Import StringIO as well

def run():
    st.subheader("Sieve Analysis (IS 2720 Part 4)")

    # Define standard sieve sizes in mm. 'Pan' represents 0.0mm.
    # These are typically in descending order for input.
    sieve_sizes = [4.75, 2.36, 1.18, 0.6, 0.425, 0.3, 0.15, 0.075, 0.0]
    sieve_labels = [str(s) if s != 0.0 else "Pan" for s in sieve_sizes]

    # Session state for input weights to retain values on rerun
    if "sieve_weights" not in st.session_state:
        st.session_state.sieve_weights = [0.0] * len(sieve_sizes)

    st.markdown("### Enter Weight Retained on Each Sieve (in grams)")

    cols = st.columns(2)
    weights_input = []
    for i, label in enumerate(sieve_labels):
        with (cols[0] if i < len(sieve_labels) / 2 else cols[1]):
            weight = st.number_input(
                f"Weight Retained on {label} mm Sieve",
                key=f"sieve_weight_input_{i}",
                min_value=0.0,
                step=0.1,
                value=st.session_state.sieve_weights[i]
            )
            weights_input.append(weight)

    st.session_state.sieve_weights = weights_input

    # Optional Save Button to download the current inputs
    if st.button("💾 Save Inputs", key="save_inputs_button"):
        input_df = pd.DataFrame({
            "Sieve Size (mm)": sieve_labels, # Use sieve_labels for "Pan" instead of 0.0
            "Weight Retained (g)": weights_input
        })

        buffer = StringIO()
        input_df.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="sieve_inputs.csv",
            mime="text/csv"
        )


    # Initialize variables for the return dictionary with default "N/A"
    soil_type = "N/A"
    suitability = "N/A"
    classification_statement = "" # Will capture the specific soil classification message
    d10_val = "N/A"
    d30_val = "N/A"
    d60_val = "N/A"
    cu_val = "N/A"
    cc_val = "N/A"
    graph_buffer = None # To store the plot image for reports

    if st.button("Calculate Sieve Analysis Results and Plot", key="sieve_analysis_calculate_button"):
        total_weight = sum(st.session_state.sieve_weights)

        if total_weight == 0:
            st.error("Please enter non-zero weights for at least one sieve to perform the calculation.")
            return None

        # Calculate % Retained and Cumulative % Passing
        percent_retained = [(w / total_weight) * 100 for w in st.session_state.sieve_weights]
        cumulative_retained = pd.Series(percent_retained).cumsum().tolist()
        percent_passing = [100 - c for c in cumulative_retained]

        # Ensure % Passing for 'Pan' (0.0 mm sieve) is explicitly 0
        if sieve_labels[-1] == "Pan":
            percent_passing[-1] = 0.0

        # Create results DataFrame for display and reporting
        results_df = pd.DataFrame({
            "S.No": list(range(1, len(sieve_labels) + 1)),
            "Sieve Size (mm)": sieve_labels,
            "Weight Retained (g)": [round(w, 2) for w in st.session_state.sieve_weights],
            "% Weight Retained": [round(p, 2) for p in percent_retained],
            "Cumulative % Retained": [round(c, 2) for c in cumulative_retained],
            "% Passing": [round(p, 2) for p in percent_passing],
        })

        st.success("✅ Sieve Analysis Results Calculated!")
        st.dataframe(results_df, use_container_width=True)

        # --- Generate Semilog Graph (Grain Size Distribution Curve) ---
        st.markdown("### Grain Size Distribution Curve")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out "Pan" (0.0mm) for plotting as log(0) is undefined.
        plot_sieve_sizes = [s for s in sieve_sizes if s > 0.0]
        # plot_percent_passing corresponds to plot_sieve_sizes.
        # This list's values will typically be decreasing as sieve_sizes decrease (finer particles).
        plot_percent_passing_for_graph = percent_passing[:-1]

        ax.semilogx(plot_sieve_sizes, plot_percent_passing_for_graph, marker='o', linestyle='-')
        ax.set_xlabel("Sieve Size (mm) - Log Scale")
        ax.set_ylabel("Percentage Finer (%)")
        ax.set_title("Grain Size Distribution Curve")
        ax.grid(True, which="both", ls="-", color='0.70') # Both major and minor grid lines
        ax.set_xlim([0.01, 100]) # Set typical x-axis limits for soil
        ax.set_ylim([0, 100]) # Percentage goes from 0 to 100

        # Customizing x-axis ticks for readability on a log scale
        major_ticks_values = np.array([0.01, 0.075, 0.15, 0.3, 0.425, 0.6, 1.18, 2.36, 4.75, 10.0, 20.0, 40.0, 80.0, 100.0])
        # Filter ticks that fall within the set x-limits
        display_ticks = [t for t in major_ticks_values if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        
        # Set the ticks and labels first
        ax.set_xticks(display_ticks)
        ax.set_xticklabels([f"{x:.3f}" if x < 1 else f"{int(x)}" for x in display_ticks])
        
        # Now invert the axis. This will flip the labels too, creating the standard geotechnical plot.
        ax.invert_xaxis()

        st.pyplot(fig) # Display the plot in Streamlit

        # Save the plot to a BytesIO object for reports
        graph_buffer = BytesIO()
        plt.savefig(graph_buffer, format="png", bbox_inches="tight")
        graph_buffer.seek(0) # Rewind buffer to the beginning
        plt.close(fig) # Close the matplotlib figure to free memory

        # --- Soil Type Classification and Suitability (IS 1498:1970) ---
        st.markdown("### Soil Type Classification (IS 1498:1970)")

        passing_475 = results_df.loc[results_df["Sieve Size (mm)"] == "4.75", "% Passing"].iloc[0] if not results_df[results_df["Sieve Size (mm)"] == "4.75"].empty else 100
        passing_075 = results_df.loc[results_df["Sieve Size (mm)"] == "0.075", "% Passing"].iloc[0] if not results_df[results_df["Sieve Size (mm)"] == "0.075"].empty else 0

        # --- Calculate D10, D30, D60, Cu, Cc ---
        # CRUCIAL FIX: For np.interp, the 'xp' (percentages) MUST be strictly ASCENDING,
        # and 'fp' (sieve sizes) must correspond and also be ASCENDING.
        
        interp_data_points = []
        for i in range(len(plot_sieve_sizes)):
            interp_data_points.append((plot_percent_passing_for_graph[i], plot_sieve_sizes[i]))

        # Sort these points based on percentage_finer (ascending order)
        # This is the crucial step to ensure correct interpolation.
        interp_data_points.sort(key=lambda x: x[0])

        # Extract sorted percentages and corresponding sieve sizes for interpolation
        passing_for_interp = [point[0] for point in interp_data_points]
        sieve_for_interp = [point[1] for point in interp_data_points]

        if not passing_for_interp or len(passing_for_interp) < 2:
            st.warning("Not enough distinct data points to reliably calculate D10, D30, D60 for interpolation.")
            d10, d30, d60, Cu, Cc = None, None, None, None, None
        else:
            min_passing_data = min(passing_for_interp)
            max_passing_data = max(passing_for_interp)

            # Perform interpolation for D10, D30, D60
            d10 = None
            if 10 >= min_passing_data and 10 <= max_passing_data:
                d10 = np.interp(10, passing_for_interp, sieve_for_interp)
            
            d30 = None
            if 30 >= min_passing_data and 30 <= max_passing_data:
                d30 = np.interp(30, passing_for_interp, sieve_for_interp)
            
            d60 = None
            if 60 >= min_passing_data and 60 <= max_passing_data:
                d60 = np.interp(60, passing_for_interp, sieve_for_interp)
            
            # Calculate Cu and Cc
            Cu = None
            if d10 is not None and d60 is not None and d10 != 0:
                Cu = d60 / d10

            Cc = None
            if d10 is not None and d30 is not None and d60 is not None and d10 != 0 and d60 != 0:
                Cc = (d30**2) / (d60 * d10)

        # Update values for display and return dictionary, handling N/A cases
        d10_val = f"{d10:.3f}" if d10 is not None else "N/A (out of range/not enough data)"
        d30_val = f"{d30:.3f}" if d30 is not None else "N/A (out of range/not enough data)"
        d60_val = f"{d60:.3f}" if d60 is not None else "N/A (out of range/not enough data)"
        cu_val = f"{Cu:.2f}" if Cu is not None else "N/A"
        cc_val = f"{Cc:.2f}" if Cc is not None else "N/A"

        st.subheader("Characteristic Diameters and Coefficients")
        st.write(f"**D10 (Effective Size):** {d10_val} mm")
        st.write(f"**D30 (Coefficient of Gradation):** {d30_val} mm")
        st.write(f"**D60 (Coefficient of Uniformity):** {d60_val} mm")
        st.write(f"**Coefficient of Uniformity (Cu):** {cu_val}")
        st.write(f"**Coefficient of Curvature (Cc):** {cc_val}")


        # --- Soil Classification Logic ---
        specific_classification_message = "" # Initialize for this run

        if passing_075 > 50:
            soil_type = "Fine-Grained Soil"
            specific_classification_message = "The soil is classified as Fine-Grained Soil. For a complete classification (Silts, Clays), Atterberg Limits are required."
            st.write("Based on **IS 1498:1970**, since more than 50% of the soil passes the 75 micron (0.075 mm) sieve, it is classified as **Fine-Grained Soil**.")
            st.warning("For a complete classification of fine-grained soils (Silts, Clays), Atterberg Limits (Liquid Limit, Plastic Limit) are required. Please perform those tests for a definitive classification (ML, MH, CL, CH etc.).")
            suitability = "Suitability for fine-grained soils varies greatly with consistency limits. Generally, clays can be problematic due to swelling/shrinkage, low strength, and poor drainage. Silts can be susceptible to frost heave and compressibility issues."
            classification_statement = specific_classification_message # Assign for the report
        else:
            soil_type = "Coarse-Grained Soil"
            st.write("Based on **IS 1498:1970**, since 50% or more of the soil is retained on the 75 micron (0.075 mm) sieve, it is classified as **Coarse-Grained Soil**.")

            gravel_fraction = 100 - passing_475 # % retained on 4.75mm sieve (material > 4.75mm)
            sand_fraction = passing_475 - passing_075 # % passing 4.75mm and retained on 0.075mm (material between 4.75mm and 0.075mm)

            if gravel_fraction > sand_fraction: # More than 50% of coarse fraction is gravel
                st.write(f"({gravel_fraction:.2f}% Gravel vs {sand_fraction:.2f}% Sand). Predominantly **Gravel (G)**")
                if passing_075 <= 5:
                    if Cu is not None and Cc is not None and Cu >= 4 and 1 <= Cc <= 3:
                        specific_classification_message = "The soil is **Well-Graded Gravel (GW)**"
                        st.info(specific_classification_message)
                        suitability = "Excellent for foundations, road bases, and drainage. High strength, low compressibility, good permeability."
                    else:
                        specific_classification_message = "The soil is **Poorly-Graded Gravel (GP)**"
                        st.info(specific_classification_message)
                        suitability = "Good for foundations and fills, but may require compaction. Lower permeability than GW, potentially more compressible. May be susceptible to segregation if poorly graded."
                    classification_statement = specific_classification_message
                elif 5 < passing_075 <= 12:
                    st.warning(f"Percentage of fines is between 5% and 12% ({passing_075:.2f}%). Dual symbol (e.g., GW-GM, GP-GC) required.")
                    specific_classification_message = "Potential classification: Dual symbol (e.g., GW-GM, GP-GC) depending on Atterberg Limits (Plasticity Index). Requires further testing."
                    st.info("Atterberg limits are needed to determine if fines are silty (M) or clayey (C).")
                    suitability = "Suitability depends on the nature of fines. Can be good for sub-base or fill, but fines might reduce permeability and strength. Requires careful evaluation."
                    classification_statement = specific_classification_message
                else: # passing_075 > 12
                    st.info(f"Percentage of fines is more than 12% ({passing_075:.2f}%).")
                    specific_classification_message = "Potential classification: **Silty Gravel (GM)** or **Clayey Gravel (GC)** (requires Atterberg limits)."
                    st.info(specific_classification_message)
                    suitability = "Suitability can be fair to poor depending on the nature and plasticity of fines. Fines can significantly reduce strength and permeability and might make the soil frost susceptible or prone to volume change."
                    classification_statement = specific_classification_message

            else: # More than 50% of coarse fraction is sand
                st.write(f"({sand_fraction:.2f}% Sand vs {gravel_fraction:.2f}% Gravel). Predominantly **Sand (S)**")
                if passing_075 <= 5:
                    if Cu is not None and Cc is not None and Cu >= 6 and 1 <= Cc <= 3:
                        specific_classification_message = "The soil is **Well-Graded Sand (SW)**"
                        st.info(specific_classification_message)
                        suitability = "Excellent for foundations, concrete aggregate, and filter materials. High strength, good drainage."
                    else:
                        specific_classification_message = "The soil is **Poorly-Graded Sand (SP)**"
                        st.info(specific_classification_message)
                        suitability = "Good for general fill and backfill, but may require more compaction. Can be susceptible to liquefaction if loose and saturated. Lower permeability than SW."
                    classification_statement = specific_classification_message
                elif 5 < passing_075 <= 12:
                    st.warning(f"Percentage of fines is between 5% and 12% ({passing_075:.2f}%). Dual symbol (e.g., SW-SM, SP-SC) required.")
                    specific_classification_message = "Potential classification: Dual symbol (e.g., SW-SM, SP-SC) depending on Atterberg Limits (Plasticity Index). Requires further testing."
                    st.info("Atterberg limits are needed to determine if fines are silty (M) or clayey (C).")
                    suitability = "Suitability depends on the nature of fines. Can be good for sub-base or fill, but fines might reduce permeability and strength. Requires careful evaluation."
                    classification_statement = specific_classification_message
                else: # passing_075 > 12
                    st.info(f"Percentage of fines is more than 12% ({passing_075:.2f}%).")
                    specific_classification_message = "Potential classification: **Silty Sand (SM)** or **Clayey Sand (SC)** (requires Atterberg limits)."
                    st.info(specific_classification_message)
                    suitability = "Suitability can be fair to poor depending on the nature and plasticity of fines. Fines can significantly reduce strength and permeability and might make the soil frost susceptible or prone to volume change."
                    classification_statement = specific_classification_message


        st.markdown("### General Suitability Remarks")
        st.info(suitability)
        st.markdown("---")
        st.markdown("Disclaimer: This classification is based solely on grain size distribution. For a complete and accurate classification as per IS 1498:1970, especially for fine-grained soils or coarse-grained soils with significant fines, Atterberg limits (Liquid Limit and Plastic Limit) are essential.")

        return {
            "Sieve Analysis Data": results_df,
            "Soil Type": soil_type,
            "Classification Detail": classification_statement, # The specific classification statement for the report
            "Suitability Remarks": suitability,
            "D10 (mm)": d10_val,
            "D30 (mm)": d30_val,
            "D60 (mm)": d60_val,
            "Cu": cu_val,
            "Cc": cc_val,
            "Graph_Image": graph_buffer
        }

    return None