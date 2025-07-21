import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import math
from docx import Document
from docx.shared import Inches

def run():
    st.subheader("📊 Consolidation Test (IS 2720 Part 15: 1986)")
    st.markdown("""
    This test determines the **magnitude and rate of consolidation** of a soil sample
    when subjected to a series of vertical loads.
    """)

    # --- Session State Initialization for Inputs ---
    if "con_initial_data" not in st.session_state:
        st.session_state.con_initial_data = {
            "initial_height": 2.0,  # cm
            "initial_dry_weight": 0.1, # g - Changed from 0.0 to 0.1 to satisfy min_value constraint
            "sample_diameter": 6.0, # cm - Typical size
            "specific_gravity": 2.65 # G_s, typical value
        }
    if "con_load_increments" not in st.session_state:
        st.session_state.con_load_increments = [] # List of dicts for each increment

    # --- Initial Sample Data ---
    st.markdown("### 🔬 Initial Sample Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.con_initial_data["initial_height"] = st.number_input(
            "Initial Height of Sample (cm)",
            min_value=0.1,
            value=st.session_state.con_initial_data["initial_height"],
            key="con_initial_height"
        )
        st.session_state.con_initial_data["initial_dry_weight"] = st.number_input(
            "Initial Dry Weight of Sample (g)",
            min_value=0.1,
            value=st.session_state.con_initial_data["initial_dry_weight"],
            key="con_initial_dry_weight"
        )
    with col2:
        st.session_state.con_initial_data["sample_diameter"] = st.number_input(
            "Diameter of Sample (cm)",
            min_value=0.1,
            value=st.session_state.con_initial_data["sample_diameter"],
            key="con_sample_diameter"
        )
        st.session_state.con_initial_data["specific_gravity"] = st.number_input(
            "Specific Gravity of Solids (Gs)",
            min_value=1.0, max_value=4.0, format="%.2f",
            value=st.session_state.con_initial_data["specific_gravity"],
            key="con_specific_gravity"
        )
    
    sample_area = (math.pi / 4) * (st.session_state.con_initial_data["sample_diameter"] ** 2)
    st.info(f"Calculated Sample Area: {sample_area:.2f} cm²")


    # --- Load Increments Data ---
    st.markdown("### 📈 Load Increments and Readings")
    st.markdown("Enter applied pressure (kg/cm²) and corresponding final dial gauge readings (mm).")
    st.markdown("Add more rows for additional load increments.")

    # Convert initial load increments to DataFrame for easier editing
    initial_load_df = pd.DataFrame(st.session_state.con_load_increments)
    if initial_load_df.empty:
        initial_load_df = pd.DataFrame({
            "Applied Pressure (kg/cm²)": [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 0.0, 0.0], # Example standard loads
            "Final Dial Gauge Reading (mm)": [0.0] * 11
        })
    
    edited_load_df = st.data_editor(
        initial_load_df,
        num_rows="dynamic",
        use_container_width=True,
        key="con_load_data_editor"
    )
    st.session_state.con_load_increments = edited_load_df.to_dict('records')

    # --- Save Inputs Button ---
    if st.button("💾 Save Inputs", key="save_con_inputs_button"):
        input_data_for_save = {
            "Initial Parameters": st.session_state.con_initial_data,
            "Load Increment Data": st.session_state.con_load_increments
        }
        
        # Convert dictionary to a string for simple CSV saving (or more complex structure if needed)
        buffer = StringIO()
        buffer.write("--- Initial Sample Parameters ---\n")
        for key, value in st.session_state.con_initial_data.items():
            buffer.write(f"{key},{value}\n")
        buffer.write("\n--- Load Increment Data ---\n")
        pd.DataFrame(st.session_state.con_load_increments).to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Input Data as CSV",
            data=buffer.getvalue(),
            file_name="consolidation_inputs.csv",
            mime="text/csv"
        )

    # --- Calculate Results Button ---
    if st.button("Calculate Consolidation Results", key="calculate_con_results_button"):
        initial_h = st.session_state.con_initial_data["initial_height"]
        initial_dry_w = st.session_state.con_initial_data["initial_dry_weight"]
        sample_dia = st.session_state.con_initial_data["sample_diameter"]
        Gs = st.session_state.con_initial_data["specific_gravity"]

        if not (initial_h > 0 and initial_dry_w > 0 and sample_dia > 0 and Gs > 0):
            st.error("Please enter valid positive values for all initial sample parameters.")
            return None

        # Calculate initial parameters
        sample_area = (math.pi / 4) * (sample_dia ** 2)
        # Unit weight of water (g/cm³)
        gamma_w = 1.0
        # Height of solids (Hs)
        Hs = initial_dry_w / (Gs * gamma_w * sample_area)
        # Initial void ratio (e0)
        e0 = (initial_h - Hs) / Hs

        # Prepare data for calculations
        calc_df_data = []
        current_void_ratio = e0
        # Assuming initial dial gauge reading is 0 for relative change calculation if not provided
        # Or take the first reading as the reference for subsequent changes
        reference_dial_gauge = None

        for i, row in edited_load_df.iterrows():
            pressure = row["Applied Pressure (kg/cm²)"]
            dial_reading = row["Final Dial Gauge Reading (mm)"]

            if i == 0:
                if pressure != 0:
                    st.warning("First load increment should ideally be 0 kg/cm² or reference starting condition.")
                reference_dial_gauge = dial_reading
                prev_pressure = 0.0
                prev_void_ratio = e0
                # Append initial state to results_df
                calc_df_data.append({
                    "Pressure (kg/cm²)": pressure,
                    "Log Pressure": math.log10(pressure) if pressure > 0 else -float('inf'),
                    "Dial Gauge Reading (mm)": dial_reading,
                    "Height (cm)": initial_h, # Current height is initial height
                    "Void Ratio (e)": round(e0, 4),
                    "Coefficient of Compressibility (av cm²/kg)": 0.0, # Not applicable for initial state
                    "Coefficient of Volume Change (mv cm²/kg)": 0.0 # Not applicable for initial state
                })
            else:
                if reference_dial_gauge is None:
                    st.error("Cannot calculate without a reference dial gauge reading (e.g., from 0 load).")
                    return None

                # Delta H in cm (assuming dial gauge in mm)
                delta_h = (reference_dial_gauge - dial_reading) / 10
                # Delta e
                delta_e = (delta_h / Hs)
                
                # Void ratio at end of increment
                current_void_ratio = e0 - delta_e

                # Delta effective stress (assuming initial stress is 0 for first loaded increment)
                delta_sigma_bar = pressure - prev_pressure

                # Coefficient of Compressibility (av)
                av = 0.0
                if delta_sigma_bar > 0:
                    av = (prev_void_ratio - current_void_ratio) / delta_sigma_bar
                
                # Coefficient of Volume Change (mv)
                mv = av / (1 + e0) # Using initial void ratio for mv calculation as per some conventions

                calc_df_data.append({
                    "Pressure (kg/cm²)": pressure,
                    "Log Pressure": math.log10(pressure) if pressure > 0 else -float('inf'),
                    "Dial Gauge Reading (mm)": dial_reading,
                    "Height (cm)": initial_h - delta_h, # Current height
                    "Void Ratio (e)": round(current_void_ratio, 4),
                    "Coefficient of Compressibility (av cm²/kg)": round(av, 5),
                    "Coefficient of Volume Change (mv cm²/kg)": round(mv, 5)
                })
                prev_pressure = pressure
                prev_void_ratio = current_void_ratio

        if not calc_df_data:
            st.error("No valid load increment data for calculations. Please ensure pressures are increasing and dial readings are valid.")
            return None

        results_df = pd.DataFrame(calc_df_data)
        st.markdown("### 📋 Consolidation Calculation Results")
        st.dataframe(results_df, use_container_width=True)

        # --- Plotting e-log(p) curve ---
        st.markdown("### 📈 Void Ratio (e) vs. Log Pressure Curve")
        fig_e_log_p, ax_e_log_p = plt.subplots(figsize=(10, 6))
        
        # Filter out invalid log pressures (-inf from 0 pressure) for plotting
        plot_df = results_df[results_df["Log Pressure"] != -float('inf')].copy()

        ax_e_log_p.plot(plot_df["Log Pressure"], plot_df["Void Ratio (e)"], marker='o', linestyle='-', color='blue')
        ax_e_log_p.set_xlabel(r"Log Pressure ($\log_{10}\bar{\sigma}$)")
        ax_e_log_p.set_ylabel("Void Ratio (e)")
        ax_e_log_p.set_title("e - log $\\bar{\sigma}$ Curve")
        ax_e_log_p.grid(True, which="both", ls="--", color='0.7')
        
        # Add labels for specific points
        for index, row in plot_df.iterrows():
            ax_e_log_p.text(row["Log Pressure"], row["Void Ratio (e)"], f'({row["Pressure (kg/cm²)"]:.1f})', fontsize=8, ha='right')

        st.pyplot(fig_e_log_p)

        img_buf_e_log_p = BytesIO()
        fig_e_log_p.savefig(img_buf_e_log_p, format="png", bbox_inches="tight")
        img_buf_e_log_p.seek(0)
        plt.close(fig_e_log_p)

        # --- Determine Compression Index (Cc) and Pre-consolidation Pressure (sigma_c_bar) ---
        st.markdown("### 🔍 Compression Index ($C_c$) and Pre-consolidation Pressure ($\bar{\sigma}_c$)")
        st.info("To determine $C_c$ and $\\bar{\\sigma}_c$, manually identify the steepest straight-line portion of the e-log p curve for $C_c$ and locate the maximum curvature for $\\bar{\\sigma}_c$. For accurate results, a more detailed graphical procedure is required.")
        
        # Simple estimation for Cc (slope of the steepest part, could be refined)
        # Find the steepest slope in the plot_df
        Cc_val = "N/A"
        if len(plot_df) >= 2:
            max_slope = 0
            # Iterating through consecutive points
            for i in range(1, len(plot_df)):
                delta_e_abs = abs(plot_df.iloc[i]["Void Ratio (e)"] - plot_df.iloc[i-1]["Void Ratio (e)"])
                delta_log_p = abs(plot_df.iloc[i]["Log Pressure"] - plot_df.iloc[i-1]["Log Pressure"])
                if delta_log_p > 0:
                    current_slope = delta_e_abs / delta_log_p
                    if current_slope > max_slope:
                        max_slope = current_slope
                        Cc_val = round(max_slope, 3)
            if Cc_val != "N/A":
                st.success(f"**Estimated Compression Index ($C_c$)**: {Cc_val}")
            else:
                st.warning("Could not estimate Compression Index. Ensure sufficient data points with varying pressures.")
        else:
            st.warning("Not enough data points to estimate Compression Index.")
        
        # Pre-consolidation pressure is hard to automate accurately, usually graphical
        st.info("Pre-consolidation Pressure ($\\bar{\\sigma}_c$): Typically determined graphically from the point of maximum curvature on the e-log p curve. Manual interpretation is usually required for accuracy.")
        sigma_c_bar_val = "Requires graphical interpretation"
        
        # --- Plotting mv-log(p) curve ---
        st.markdown("### 📈 Coefficient of Volume Change ($m_v$) vs. Log Pressure Curve")
        fig_mv_log_p, ax_mv_log_p = plt.subplots(figsize=(10, 6))
        
        # Filter out invalid log pressures and mv for plotting
        plot_df_mv = results_df[(results_df["Log Pressure"] != -float('inf')) & (results_df["Coefficient of Volume Change (mv cm²/kg)"] >= 0)].copy()

        if not plot_df_mv.empty:
            ax_mv_log_p.plot(plot_df_mv["Log Pressure"], plot_df_mv["Coefficient of Volume Change (mv cm²/kg)"], marker='o', linestyle='-', color='purple')
            ax_mv_log_p.set_xlabel(r"Log Pressure ($\log_{10}\bar{\sigma}$)")
            ax_mv_log_p.set_ylabel("Coefficient of Volume Change ($m_v$ cm²/kg)")
            ax_mv_log_p.set_title("$m_v$ - log $\\bar{\sigma}$ Curve")
            ax_mv_log_p.grid(True, which="both", ls="--", color='0.7')
            st.pyplot(fig_mv_log_p)

            img_buf_mv_log_p = BytesIO()
            fig_mv_log_p.savefig(img_buf_mv_log_p, format="png", bbox_inches="tight")
            img_buf_mv_log_p.seek(0)
            plt.close(fig_mv_log_p)
        else:
            st.warning("Not enough valid data points to plot $m_v$ - log $\\bar{\sigma}$ curve.")
            img_buf_mv_log_p = None

        # --- Summarize Soil Properties based on Consolidation ---
        st.markdown("### 📝 General Remarks on Compressibility")
        if Cc_val != "N/A" and isinstance(Cc_val, (int, float)):
            if Cc_val < 0.2:
                st.info("The soil exhibits **low compressibility**.")
                compressibility_remarks = "Low compressibility."
            elif 0.2 <= Cc_val <= 0.4:
                st.info("The soil exhibits **medium compressibility**.")
                compressibility_remarks = "Medium compressibility."
            else:
                st.warning("The soil exhibits **high compressibility**.")
                compressibility_remarks = "High compressibility."
        else:
            compressibility_remarks = "Compressibility remarks require Compression Index (Cc) calculation."
            st.info(compressibility_remarks)

        # --- Report Generation for this specific test ---
        st.markdown("---")
        st.markdown("### 📄 Generate Report for Consolidation Test")
        report_format = st.radio("Select report format for this test", ("Excel", "Word (DOCX)"), key="con_report_format")

        if st.button("📥 Download Report for this Test", key="con_download_report"):
            if report_format == "Excel":
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Initial Parameters
                    initial_params_df = pd.DataFrame({
                        "Parameter": ["Initial Height (cm)", "Initial Dry Weight (g)", "Sample Diameter (cm)", "Specific Gravity (Gs)", "Sample Area (cm²)", "Height of Solids (Hs)", "Initial Void Ratio (e0)"],
                        "Value": [initial_h, initial_dry_w, sample_dia, Gs, round(sample_area, 2), round(Hs, 4), round(e0, 4)]
                    })
                    initial_params_df.to_excel(writer, sheet_name="Initial Parameters", index=False)

                    # Calculated Consolidation Data
                    results_df.to_excel(writer, sheet_name="Consolidation Data", index=False)

                    # Summary of Parameters
                    summary_params_df = pd.DataFrame({
                        "Parameter": ["Estimated Compression Index (Cc)", "Pre-consolidation Pressure (sigma_c_bar)", "Compressibility Remarks"],
                        "Value": [str(Cc_val), sigma_c_bar_val, compressibility_remarks]
                    })
                    summary_params_df.to_excel(writer, sheet_name="Summary Parameters", index=False)

                output.seek(0)
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name="Consolidation_Test_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            elif report_format == "Word (DOCX)":
                doc = Document()
                doc.add_heading("Consolidation Test Report", level=0)

                # Initial Parameters
                doc.add_heading("1. Initial Sample Parameters", level=1)
                initial_params_df = pd.DataFrame({
                    "Parameter": ["Initial Height (cm)", "Initial Dry Weight (g)", "Sample Diameter (cm)", "Specific Gravity (Gs)", "Sample Area (cm²)", "Height of Solids (Hs)", "Initial Void Ratio (e0)"],
                    "Value": [initial_h, initial_dry_w, sample_dia, Gs, round(sample_area, 2), round(Hs, 4), round(e0, 4)]
                })
                table = doc.add_table(rows=1, cols=len(initial_params_df.columns))
                table.style = 'Table Grid'
                hdr_cells = table.rows[0].cells
                for i, col in enumerate(initial_params_df.columns):
                    hdr_cells[i].text = str(col)
                for _, row in initial_params_df.iterrows():
                    row_cells = table.add_row().cells
                    for i, val in enumerate(row):
                        row_cells[i].text = str(val)
                doc.add_paragraph("")

                # Calculated Consolidation Data
                doc.add_heading("2. Consolidation Calculation Results", level=1)
                table = doc.add_table(rows=1, cols=len(results_df.columns))
                table.style = 'Table Grid'
                hdr_cells = table.rows[0].cells
                for i, col in enumerate(results_df.columns):
                    hdr_cells[i].text = str(col)
                for _, row in results_df.iterrows():
                    row_cells = table.add_row().cells
                    for i, val in enumerate(row):
                        row_cells[i].text = str(val)
                doc.add_paragraph("")

                # e-log(p) Curve
                if img_buf_e_log_p:
                    doc.add_heading("3. Void Ratio (e) vs. Log Pressure Curve", level=1)
                    try:
                        img_buf_e_log_p.seek(0)
                        doc.add_picture(img_buf_e_log_p, width=Inches(5))
                        doc.add_paragraph("")
                    except Exception as e:
                        doc.add_paragraph(f"Could not load e-log(p) curve: {e}")

                # mv-log(p) Curve
                if img_buf_mv_log_p:
                    doc.add_heading("4. Coefficient of Volume Change ($m_v$) vs. Log Pressure Curve", level=1)
                    try:
                        img_buf_mv_log_p.seek(0)
                        doc.add_picture(img_buf_mv_log_p, width=Inches(5))
                        doc.add_paragraph("")
                    except Exception as e:
                        doc.add_paragraph(f"Could not load mv-log(p) curve: {e}")

                # Compression Index and Pre-consolidation Pressure
                doc.add_heading("5. Compression Index and Pre-consolidation Pressure", level=1)
                # FIX: Replaced problematic f-string with a raw string for LaTeX
                doc.add_paragraph(f"**Estimated Compression Index ($C_c$)**: {Cc_val}")
                doc.add_paragraph(f"**Pre-consolidation Pressure ($\\bar{{\\sigma}}_c$)**: {sigma_c_bar_val}")
                doc.add_paragraph("")

                # General Remarks on Compressibility
                doc.add_heading("6. General Remarks on Compressibility", level=1)
                doc.add_paragraph(compressibility_remarks)

                buffer = BytesIO()
                doc.save(buffer)
                st.download_button(
                    label="📥 Download Word Report",
                    data=buffer.getvalue(),
                    file_name="Consolidation_Test_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    return {
        "Initial Parameters": pd.DataFrame({
            "Parameter": ["Initial Height (cm)", "Initial Dry Weight (g)", "Sample Diameter (cm)", "Specific Gravity (Gs)", "Sample Area (cm²)"],
            "Value": [initial_h if 'initial_h' in locals() else 'N/A',
                      initial_dry_w if 'initial_dry_w' in locals() else 'N/A',
                      sample_dia if 'sample_dia' in locals() else 'N/A',
                      Gs if 'Gs' in locals() else 'N/A',
                      round(sample_area, 2) if 'sample_area' in locals() else 'N/A']
        }),
        "Calculated Consolidation Data": results_df if 'results_df' in locals() else pd.DataFrame(),
        "e-log(p) Curve": img_buf_e_log_p if 'img_buf_e_log_p' in locals() else None,
        "mv-log(p) Curve": img_buf_mv_log_p if 'img_buf_mv_log_p' in locals() else None,
        "Estimated Compression Index (Cc)": str(Cc_val) if 'Cc_val' in locals() else "N/A",
        "Pre-consolidation Pressure (sigma_c_bar)": sigma_c_bar_val if 'sigma_c_bar_val' in locals() else "N/A",
        "Compressibility Remarks": compressibility_remarks if 'compressibility_remarks' in locals() else "",
        "Remarks": "Consolidation characteristics determined from oedometer test, including void ratio, coefficients of compressibility and volume change."
    }
