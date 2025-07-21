import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import math
from io import BytesIO, StringIO # Import StringIO for CSV buffer

def run():
    st.subheader("🧪 Light Compaction Test (IS 2720 Part 7:1980)")

    st.markdown("""
    This test determines the **Optimum Moisture Content (OMC)** and **Maximum Dry Density (MDD)** using the standard Proctor method.

    **Required Inputs Per Trial:**
    - W1: Weight of empty cup
    - W2: Weight of cup + wet soil
    - W3: Weight of cup + dry soil
    - W4: Weight of mould + base plate
    - W5: Weight of mould + compacted soil + base plate
    """)

    # --- Session State Initialization for Inputs ---
    # Initialize mould dimensions in session state
    if "lc_mould_dia" not in st.session_state:
        st.session_state.lc_mould_dia = 10.0
    if "lc_mould_height" not in st.session_state:
        st.session_state.lc_mould_height = 12.7
    
    # Initialize number of trials in session state
    if "lc_num_points" not in st.session_state:
        st.session_state.lc_num_points = 5

    # Number of trials input
    num_points = st.number_input(
        "🔢 Number of Compaction Trials",
        min_value=3, max_value=10,
        value=st.session_state.lc_num_points,
        step=1,
        key="lc_num_points_input"
    )

    # Update session state if num_points changes and reinitialize trial inputs
    if num_points != st.session_state.lc_num_points:
        st.session_state.lc_num_points = num_points
        # Reinitialize lc_trial_inputs if num_points changes
        st.session_state.lc_trial_inputs = [
            {"W1": 0.0, "W2": 0.0, "W3": 0.0, "W4": 0.0, "W5": 0.0}
            for _ in range(num_points)
        ]

    # Initialize trial inputs in session state
    if "lc_trial_inputs" not in st.session_state:
        st.session_state.lc_trial_inputs = [
            {"W1": 0.0, "W2": 0.0, "W3": 0.0, "W4": 0.0, "W5": 0.0}
            for _ in range(num_points)
        ]
    # Ensure lc_trial_inputs list size matches current num_points
    while len(st.session_state.lc_trial_inputs) < num_points:
        st.session_state.lc_trial_inputs.append({"W1": 0.0, "W2": 0.0, "W3": 0.0, "W4": 0.0, "W5": 0.0})
    st.session_state.lc_trial_inputs = st.session_state.lc_trial_inputs[:num_points]


    # Mould size inputs, bound to session state
    st.markdown("### 📏 Mould Dimensions")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.lc_mould_dia = st.number_input(
            "Diameter (cm)", value=st.session_state.lc_mould_dia, min_value=1.0, key="lc_dia"
        )
    with col2:
        st.session_state.lc_mould_height = st.number_input(
            "Height (cm)", value=st.session_state.lc_mould_height, min_value=1.0, key="lc_height"
        )

    volume = (math.pi / 4) * st.session_state.lc_mould_dia**2 * st.session_state.lc_mould_height  # in cm³
    st.info(f"📐 Volume of Mould = **{volume:.2f} cm³**")

    # Trial data input, bound to session state
    st.markdown("### 📋 Enter Data for Each Trial")

    for i in range(num_points):
        st.markdown(f"#### Trial {i + 1}")
        c1, c2 = st.columns(2) # Changed to 2 columns for better layout, removing calculation column
        with c1:
            st.session_state.lc_trial_inputs[i]["W1"] = st.number_input(
                f"W1: Empty Cup (g)", min_value=0.0, key=f"lc_W1_{i}", value=st.session_state.lc_trial_inputs[i]["W1"]
            )
            st.session_state.lc_trial_inputs[i]["W2"] = st.number_input(
                f"W2: Cup + Wet Soil (g)", min_value=0.0, key=f"lc_W2_{i}", value=st.session_state.lc_trial_inputs[i]["W2"]
            )
            st.session_state.lc_trial_inputs[i]["W3"] = st.number_input(
                f"W3: Cup + Dry Soil (g)", min_value=0.0, key=f"lc_W3_{i}", value=st.session_state.lc_trial_inputs[i]["W3"]
            )
        with c2:
            st.session_state.lc_trial_inputs[i]["W4"] = st.number_input(
                f"W4: Mould + Base Plate (g)", min_value=0.0, key=f"lc_W4_{i}", value=st.session_state.lc_trial_inputs[i]["W4"]
            )
            st.session_state.lc_trial_inputs[i]["W5"] = st.number_input(
                f"W5: Mould + Soil + Plate (g)", min_value=0.0, key=f"lc_W5_{i}", value=st.session_state.lc_trial_inputs[i]["W5"]
            )
    
    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_lc_inputs_button"):
        # Create a DataFrame for inputs, adding mould dimensions
        input_df_data = []
        for i, trial_data in enumerate(st.session_state.lc_trial_inputs):
            row = {
                "Trial": i + 1,
                "W1 (g)": trial_data["W1"],
                "W2 (g)": trial_data["W2"],
                "W3 (g)": trial_data["W3"],
                "W4 (g)": trial_data["W4"],
                "W5 (g)": trial_data["W5"],
                "Mould Diameter (cm)": st.session_state.lc_mould_dia,
                "Mould Height (cm)": st.session_state.lc_mould_height
            }
            input_df_data.append(row)
        
        input_df_to_save = pd.DataFrame(input_df_data)

        buffer = StringIO()
        input_df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="light_compaction_inputs.csv",
            mime="text/csv"
        )
    
    # --- Calculate Results Button ---
    # Moved the calculation logic into a button to control execution
    if st.button("Calculate Compaction Results", key="calculate_lc_results_button"):
        calculated_data = []
        for i, trial_input in enumerate(st.session_state.lc_trial_inputs):
            W1 = trial_input["W1"]
            W2 = trial_input["W2"]
            W3 = trial_input["W3"]
            W4 = trial_input["W4"]
            W5 = trial_input["W5"]

            # Check data validity before calculating
            if W1 > 0 and W2 > W1 and W3 > W1 and W5 > W4:
                try:
                    weight_wet = W2 - W1
                    weight_dry = W3 - W1
                    water_content = ((weight_wet - weight_dry) / weight_dry) * 100
                    wet_density = (W5 - W4) / volume  # g/cm³
                    dry_density = wet_density / (1 + water_content / 100)

                    calculated_data.append({
                        "Trial": i + 1,
                        "Water Content (%)": round(water_content, 2),
                        "Wet Density (g/cc)": round(wet_density, 3),
                        "Dry Density (g/cc)": round(dry_density, 3)
                    })
                except ZeroDivisionError:
                    st.warning(f"Trial {i+1}: Cannot calculate due to zero dry weight (W3-W1). Check inputs.")
                except Exception as e:
                    st.warning(f"Trial {i+1}: Error in calculation: {e}. Check inputs.")
            else:
                st.warning(f"Trial {i+1}: Incomplete or invalid input data. Please check W1, W2, W3, W4, W5 values.")
        
        if calculated_data:
            df = pd.DataFrame(calculated_data)
            st.markdown("### 📊 Compaction Results Table")
            st.dataframe(df, use_container_width=True)

            # Find OMC and MDD from the calculated data
            mdd = df["Dry Density (g/cc)"].max()
            omc = df.loc[df["Dry Density (g/cc)"].idxmax(), "Water Content (%)"]

            st.success(f"✅ **Maximum Dry Density (MDD)** = {mdd:.3f} g/cc")
            st.success(f"✅ **Optimum Moisture Content (OMC)** = {omc:.2f}%")

            # Plot Compaction Curve
            st.markdown("### 📈 Compaction Curve")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df["Water Content (%)"], df["Dry Density (g/cc)"], marker='o', color='blue', linestyle='-')
            ax.set_xlabel("Water Content (%)")
            ax.set_ylabel("Dry Density (g/cc)")
            ax.set_title("Compaction Curve")
            ax.grid(True)
            
            # Highlight OMC and MDD on the plot
            ax.plot(omc, mdd, 'ro', markersize=8, label=f'OMC ({omc:.2f}%), MDD ({mdd:.3f} g/cc)')
            ax.vlines(omc, ax.get_ylim()[0], mdd, color='red', linestyle='--', linewidth=0.7)
            ax.hlines(mdd, ax.get_xlim()[0], omc, color='red', linestyle='--', linewidth=0.7)
            ax.legend()

            st.pyplot(fig)

            img_buf = BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            img_buf.seek(0)
            plt.close(fig) # Close the figure to free memory

            # IS Code Suitability
            suitability_remark = ""
            st.markdown("### ✅ Suitability Based on IS Code")
            if mdd > 1.8:
                suitability_remark = "Suitable for base/subgrade (high compaction)."
                st.info("🟢 " + suitability_remark)
            elif 1.6 <= mdd <= 1.8:
                suitability_remark = "Moderately compacted. Acceptable for fill layers."
                st.info("🟡 " + suitability_remark)
            else:
                suitability_remark = "Low compaction. Soil may require stabilization."
                st.warning("🔴 " + suitability_remark)
            
            # Return results for the main app to collect
            return {
                "Mould Dimensions": pd.DataFrame({"Parameter": ["Diameter (cm)", "Height (cm)", "Volume (cm³)"],
                                                  "Value": [st.session_state.lc_mould_dia, st.session_state.lc_mould_height, volume]}),
                "Test Results Data": df,
                "Maximum Dry Density (MDD)": f"{mdd:.3f} g/cc",
                "Optimum Moisture Content (OMC)": f"{omc:.2f}%",
                "Compaction Curve Graph": img_buf,
                "Suitability Remarks": suitability_remark
            }
        else:
            st.error("⚠ No valid data points to calculate MDD and OMC. Please ensure all trials have valid and complete inputs.")
            return None # Return None if no valid data processed

    return None # Default return if calculation button is not pressed