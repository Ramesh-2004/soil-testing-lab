# 🌱 Soil Testing Virtual Lab

A Streamlit-based virtual lab for conducting standard geotechnical soil tests, built according to IS codes.

## 📋 Features

- Sieve Analysis
- Liquid Limit (Casagrande)
- Plastic Limit
- Core Cutter Test
- Specific Gravity Test
- Constant Head Permeability
- Variable Head Permeability
- Light Compaction Test
- Direct Shear Test
- Unconfined Compression Test (UCS)
- Save input data and generate reports (Excel/Word)

## 🛠️ Technologies Used

- Python 🐍
- Streamlit 📈
- Pandas, NumPy
- Matplotlib
- `python-docx` for Word reports
- `openpyxl` for Excel reports

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Ramesh-2004/soil-testing-lab.git
cd soil-testing-lab

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
