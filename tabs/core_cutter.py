import streamlit as st
import numpy as np
import pandas as pd # Import pandas for DataFrame creation for saving inputs
from io import BytesIO, StringIO # Import StringIO for CSV buffer

def run():
    st.subheader("In-situ Density by Core Cutter Method (IS 2720: Part 29: 1975, reaffirmed 1995)")

    st.markdown("""
    This test determines the bulk and dry density of soil using the core cutter method.

    **Required Inputs:**
    - Core cutter dimensions (height and internal diameter)
    - Weights of cutter and soil sample
    - Weights for moisture content determination
    """)

    # --- Session State Initialization for Inputs ---
    # Initialize all input variables in session state
    if "cc_h" not in st.session_state:
        st.session_state.cc_h = 12.8
    if "cc_d" not in st.session_state:
        st.session_state.cc_d = 10.0
    if "cc_wt_empty" not in st.session_state:
        st.session_state.cc_wt_empty = 0.0
    if "cc_wt_full" not in st.session_state:
        st.session_state.cc_wt_full = 0.0
    if "cc_wt_container" not in st.session_state:
        st.session_state.cc_wt_container = 0.0
    if "cc_wt_wet" not in st.session_state:
        st.session_state.cc_wt_wet = 0.0
    if "cc_wt_dry" not in st.session_state:
        st.session_state.cc_wt_dry = 0.0

    # --- Input Fields (bound to session state) ---
    st.markdown("### 📏 Core Cutter Dimensions")
    st.session_state.cc_h = st.number_input(
        "Height of Core Cutter (cm)", min_value=0.0, value=st.session_state.cc_h, key="cc_h_input"
    )
    st.session_state.cc_d = st.number_input(
        "Internal Diameter of Core Cutter (cm)", min_value=0.0, value=st.session_state.cc_d, key="cc_d_input"
    )

    volume = (np.pi * (st.session_state.cc_d / 2) ** 2 * st.session_state.cc_h)  # in cm³
    st.info(f"📐 Calculated Volume of Core Cutter = **{volume:.2f} cm³**")

    st.markdown("### ⚖️ Weight Measurements")
    st.session_state.cc_wt_empty = st.number_input(
        "Weight of Empty Core Cutter (g)", min_value=0.0, value=st.session_state.cc_wt_empty, key="cc_wt_empty_input"
    )
    st.session_state.cc_wt_full = st.number_input(
        "Weight of Core Cutter with Soil (g)", min_value=0.0, value=st.session_state.cc_wt_full, key="cc_wt_full_input"
    )
    st.session_state.cc_wt_container = st.number_input(
        "Weight of Empty Moisture Container (g)", min_value=0.0, value=st.session_state.cc_wt_container, key="cc_wt_container_input"
    )
    st.session_state.cc_wt_wet = st.number_input(
        "Weight of Container + Wet Soil (g)", min_value=0.0, value=st.session_state.cc_wt_wet, key="cc_wt_wet_input"
    )
    st.session_state.cc_wt_dry = st.number_input(
        "Weight of Container + Dry Soil (g)", min_value=0.0, value=st.session_state.cc_wt_dry, key="cc_wt_dry_input"
    )

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_core_cutter_inputs_button"):
        input_data = {
            "Parameter": [
                "Height of Core Cutter (cm)",
                "Internal Diameter of Core Cutter (cm)",
                "Weight of Empty Core Cutter (g)",
                "Weight of Core Cutter with Soil (g)",
                "Weight of Empty Moisture Container (g)",
                "Weight of Container + Wet Soil (g)",
                "Weight of Container + Dry Soil (g)"
            ],
            "Value": [
                st.session_state.cc_h,
                st.session_state.cc_d,
                st.session_state.cc_wt_empty,
                st.session_state.cc_wt_full,
                st.session_state.cc_wt_container,
                st.session_state.cc_wt_wet,
                st.session_state.cc_wt_dry
            ]
        }
        input_df_to_save = pd.DataFrame(input_data)
        
        buffer = StringIO()
        input_df_to_save.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="core_cutter_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Button ---
    if st.button("Calculate Core Cutter Results", key="calculate_core_cutter_button"):
        # Retrieve values from session state for calculation
        h = st.session_state.cc_h
        d = st.session_state.cc_d
        wt_empty = st.session_state.cc_wt_empty
        wt_full = st.session_state.cc_wt_full
        wt_container = st.session_state.cc_wt_container
        wt_wet = st.session_state.cc_wt_wet
        wt_dry = st.session_state.cc_wt_dry

        # Validate inputs
        if not (h > 0 and d > 0 and wt_empty >= 0 and wt_full > wt_empty and wt_container >= 0 and wt_wet > wt_container and wt_dry > wt_container):
            st.error("Please enter valid and consistent positive values for all inputs to perform calculations.")
            return None

        try:
            # Volume in cm³
            volume = (np.pi * (d / 2) ** 2 * h)

            wt_soil = wt_full - wt_empty  # g
            bulk_density = wt_soil / volume  # g/cc or Mg/m³

            # Moisture content
            weight_of_dry_soil_in_container = wt_dry - wt_container
            if weight_of_dry_soil_in_container > 0:
                moisture_content = ((wt_wet - wt_dry) / weight_of_dry_soil_in_container) * 100
            else:
                moisture_content = 0.0 # Or handle as an error if 0 dry soil is not expected
                st.warning("Weight of dry soil in moisture container is zero. Moisture content cannot be calculated.")


            # Dry density
            if (1 + (moisture_content / 100)) > 0: # Avoid division by zero if moisture content is -100% (unlikely but good check)
                dry_density = bulk_density / (1 + (moisture_content / 100))
            else:
                dry_density = 0.0 # Or handle as an error
                st.warning("Calculated moisture content leads to division by zero for dry density. Check inputs.")


            # Display volume separately
            st.markdown("### Core Cutter Volume")
            st.info(f"Volume = {volume:.2f} cm³")

            # Highlighted Results
            st.markdown("### Results")
            st.info(f"**Bulk Density = {bulk_density:.2f} g/cc**")
            st.info(f"**Moisture Content = {moisture_content:.2f}%**")
            st.success(f"**Dry Density = {dry_density:.2f} g/cc**")

            # Suitability comment
            suitability_remark = ""
            st.markdown("### Suitability (as per IS 2720)")
            if dry_density < 1.4:
                suitability_remark = "Soil has **low compaction** — may not be suitable for subgrade."
                st.warning(suitability_remark)
            elif 1.4 <= dry_density < 1.75:
                suitability_remark = "**Moderately compacted** soil — can be used for general fill."
                st.info(suitability_remark)
            else:
                suitability_remark = "**Well compacted** soil — suitable for base/subbase layers."
                st.success(suitability_remark)

            # Return results for the main app to collect
            results_df = pd.DataFrame({
                "Parameter": ["Volume of Core Cutter (cm³)", "Bulk Density (g/cc)", "Moisture Content (%)", "Dry Density (g/cc)"],
                "Value": [round(volume, 2), round(bulk_density, 2), round(moisture_content, 2), round(dry_density, 2)]
            })

            return {
                "Input Data": pd.DataFrame(input_data), # Return raw inputs as a DataFrame
                "Calculated Results": results_df,
                "Suitability Remarks": suitability_remark,
                "Remarks": "In-situ density determined using the Core Cutter Method."
            }

        except Exception as e:
            st.error(f"An error occurred during calculation: {e}. Please check your input values.")
            return None
    
    return None # Default return if calculation button is not pressed