import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches
import math
import json


def run():
    st.subheader("🔗 Direct Shear Test (IS 2720 Part 13:1986)")
    st.markdown("""
    This test is performed to determine the **shear strength parameters (cohesion 'c' and
    angle of internal friction 'phi')** of a soil sample under drained conditions.
    """)

    box_dim = st.number_input("Side Length of Shear Box (mm)", min_value=1.0, value=60.0)
    area = (box_dim / 10) ** 2  # in cm²
    st.info(f"Calculated Shear Area: {area:.2f} cm²")

    proving_ring_const = st.number_input("Proving Ring Constant (kg/div)", value=1.0)
    dial_lc = st.number_input("Dial Gauge Least Count (mm/div)", value=0.01)

    num_trials = st.number_input("Number of Normal Stress Trials", min_value=1, max_value=5, value=3, step=1)

    if "trial_inputs" not in st.session_state or len(st.session_state.trial_inputs) != num_trials:
        st.session_state.trial_inputs = [{} for _ in range(num_trials)]

    input_complete = True
    normal_stresses = []
    shear_stresses = []
    all_dfs = []

    for i in range(num_trials):
        st.markdown(f"### Trial {i+1}")
        sigma_n = st.number_input(f"Normal Stress σₙ (kg/cm²) - Trial {i+1}", value=(i+1)*0.5, key=f"norm_{i}")
        n = st.number_input(f"Number of Readings - Trial {i+1}", min_value=2, value=10, key=f"num_read_{i}")

        if f"ds_data_{i}" not in st.session_state or len(st.session_state[f"ds_data_{i}"]) != n:
            st.session_state[f"ds_data_{i}"] = pd.DataFrame({
                "Horizontal Deformation (div)": [0.0] * n,
                "Proving Ring Reading (div)": [0.0] * n
            })

        df = st.data_editor(st.session_state[f"ds_data_{i}"], key=f"ds_data_editor_{i}", num_rows="dynamic")

        st.session_state.trial_inputs[i] = {
            "sigma_n": sigma_n,
            "n": n,
            "data": df.to_dict(orient="list")
        }

    if st.button("💾 Save Inputs"):
        saved = json.dumps(st.session_state.trial_inputs, indent=2)
        st.download_button("📥 Download Saved Inputs (JSON)", data=saved, file_name="direct_shear_inputs.json")

    if st.button("🧮 Calculate Results"):
        for i in range(num_trials):
            trial = st.session_state.trial_inputs[i]
            sigma_n = trial["sigma_n"]
            df = pd.DataFrame(trial["data"])

            df["Shear Force (kg)"] = df["Proving Ring Reading (div)"] * proving_ring_const
            df["Shear Stress (kg/cm²)"] = df["Shear Force (kg)"] / area
            df["Deformation (mm)"] = df["Horizontal Deformation (div)"] * dial_lc

            st.markdown(f"### 📊 Results for Trial {i+1}")
            st.dataframe(df)

            fig, ax = plt.subplots()
            ax.plot(df["Deformation (mm)"], df["Shear Stress (kg/cm²)"], marker='o')
            ax.set_title(f"Trial {i+1}: Shear Stress vs Deformation")
            ax.set_xlabel("Deformation (mm)")
            ax.set_ylabel("Shear Stress (kg/cm²)")
            ax.grid(True)
            st.pyplot(fig)

            tau_max = df["Shear Stress (kg/cm²)"].max()
            shear_stresses.append(tau_max)
            normal_stresses.append(sigma_n)
            all_dfs.append(df.copy())

        if len(shear_stresses) >= 2:
            # Mohr-Coulomb Line Fit
            coeffs = np.polyfit(normal_stresses, shear_stresses, 1)
            phi_rad = math.atan(coeffs[0])
            phi_deg = math.degrees(phi_rad)
            cohesion = coeffs[1]

            st.markdown("### ✅ Final Results (After All Trials)")
            st.write(f"Cohesion (c): **{cohesion:.3f} kg/cm²**")
            st.write(f"Angle of Internal Friction (ϕ): **{phi_deg:.2f}°**")

            fig2, ax2 = plt.subplots()
            ax2.scatter(normal_stresses, shear_stresses, color='blue', label='Data Points')
            x_line = np.linspace(0, max(normal_stresses) * 1.1, 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax2.plot(x_line, y_line, color='red', label='Mohr-Coulomb Fit')
            ax2.set_xlabel("Normal Stress σₙ (kg/cm²)")
            ax2.set_ylabel("Shear Stress τ (kg/cm²)")
            ax2.set_title("Shear Stress vs Normal Stress")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

            # Report Generation
            if st.button("📄 Generate Combined Word Report"):
                doc = Document()
                doc.add_heading("Direct Shear Test Report", 0)
                doc.add_paragraph(f"Shear Box Size: {box_dim} mm")
                doc.add_paragraph(f"Proving Ring Constant: {proving_ring_const} kg/div")
                doc.add_paragraph(f"Dial Gauge LC: {dial_lc} mm/div")
                doc.add_paragraph(f"Area: {area:.2f} cm²")
                doc.add_paragraph(f"Cohesion (c): {cohesion:.3f} kg/cm²")
                doc.add_paragraph(f"Angle of Internal Friction (ϕ): {phi_deg:.2f} degrees")

                for i in range(num_trials):
                    df = all_dfs[i]
                    doc.add_heading(f"Trial {i+1}", level=2)
                    doc.add_paragraph(f"Normal Stress: {normal_stresses[i]:.2f} kg/cm²")
                    table = doc.add_table(rows=1, cols=4)
                    hdr_cells = table.rows[0].cells
                    hdr_cells[0].text = 'Deformation (mm)'
                    hdr_cells[1].text = 'Proving Ring (div)'
                    hdr_cells[2].text = 'Shear Stress (kg/cm²)'
                    hdr_cells[3].text = 'Shear Force (kg)'
                    for idx in range(len(df)):
                        row = table.add_row().cells
                        row[0].text = f"{df['Deformation (mm)'][idx]:.2f}"
                        row[1].text = f"{df['Proving Ring Reading (div)'][idx]:.2f}"
                        row[2].text = f"{df['Shear Stress (kg/cm²)'][idx]:.2f}"
                        row[3].text = f"{df['Shear Force (kg)'][idx]:.2f}"

                buffer = BytesIO()
                doc.save(buffer)
                buffer.seek(0)
                st.download_button(
                    "📥 Download Word Report",
                    data=buffer.getvalue(),
                    file_name="Direct_Shear_Test_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
