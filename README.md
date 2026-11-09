# Patient Risk Modeling for Clinical Diagnostics

A data science project completed during the Clinical Research and Data Science Internship at the **Institute of Innovations** (October 2025).

## Overview
This project processes clinical vital signs data, performs statistical cleaning and preprocessing, categorizes patient clusters using unsupervised learning (K-Means), and predicts patient risk levels using supervised machine learning (Random Forest Classifier).

## Tech Stack
- **R / RStudio**: Data cleaning, duplicate removal, handling missing values, and column standardization.
- **Python**: Machine Learning (pandas, scikit-learn, matplotlib, numpy).
- **KNIME**: Visual workflow construction for clinical pipelines.

## Project Structure
- `app.py` - Streamlit web interface for analysis and visualizations.
- `clean_vital_signs.R` - Data preprocessing script in R. Run it locally; don't commit patient datasets.
- `patient_risk_model.py` - Machine learning modeling script in Python.
- `requirements.txt` - Python application dependencies.

## How to Run
1. Run `clean_vital_signs.R` in RStudio to preprocess the raw clinical data.
2. Install the Python dependencies with `python -m pip install -r requirements.txt`.
3. Start the web interface with `python -m streamlit run app.py`.
4. Upload the locally generated `vital_signs_clean.csv` in the browser to view clusters, risk-rule counts, feature importance, and an illustrative prediction.
5. To generate the Excel output from the command line, run `python patient_risk_model.py`.

## Deploy with GitHub and Streamlit Community Cloud
GitHub stores the source code; Streamlit Community Cloud runs the interactive app.

1. Create a GitHub repository named `patient-risk-model` under your GitHub account.
2. Push this project to that repository. The `.gitignore` excludes CSV and Excel files so patient datasets and generated reports are not uploaded.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize access to the repository.
4. Choose `app.py` as the app file and deploy.

The deployed app accepts a cleaned CSV upload and analyzes it in memory. Do not upload identifiable or sensitive patient data to a public app. Restrict access using the hosting platform's available options. The risk rule and model are educational only and are not clinically validated.
