"""Streamlit interface for the educational patient risk analysis."""

from io import BytesIO

import pandas as pd
import streamlit as st

from patient_risk_model import FEATURES, analyze_vitals

st.set_page_config(page_title="Vital Signs Analyzer", page_icon="🩺", layout="wide")
st.title("Vital Signs Analyzer")
st.warning(
    "Educational demonstration only. Risk labels use the rule "
    "`Temp > 37 °C or heart_rate > 85`; they are not medical diagnoses or "
    "clinically validated predictions. Do not upload identifiable patient data."
)
st.caption(
    "Upload a cleaned CSV containing age, bp, heart_rate, and Temp. "
    "Uploaded data is analyzed in memory and is not saved by this app."
)

uploaded_file = st.file_uploader("Cleaned vital signs CSV", type=["csv"])

if uploaded_file is not None:
    try:
        source = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
    else:
        try:
            data, cluster_summary, model, feature_importance, dropped_rows = (
                analyze_vitals(source)
            )
        except ValueError as error:
            st.error(str(error))
        else:
            if dropped_rows:
                st.info(
                    f"Excluded {dropped_rows} row(s) with missing or invalid "
                    "required numeric values."
                )

            st.subheader("Dataset preview")
            st.dataframe(data.head(20), use_container_width=True)
            st.caption(
                f"Analyzed {len(data)} of {len(source)} uploaded records."
            )

            left, right = st.columns(2)
            with left:
                st.subheader("Risk level counts")
                risk_counts = (
                    data["RiskLevel"]
                    .value_counts()
                    .reindex(["Low", "High"], fill_value=0)
                    .rename_axis("Risk level")
                    .to_frame("Patients")
                )
                st.bar_chart(risk_counts)
            with right:
                st.subheader("Cluster averages")
                st.dataframe(cluster_summary.round(2), use_container_width=True)

            st.subheader("Model feature importance")
            st.bar_chart(feature_importance.to_frame())
            st.caption(
                "The classifier is trained to reproduce the threshold rule shown "
                "above. Feature importance and predictions are illustrative, not "
                "evidence of clinical performance."
            )

            st.subheader("Illustrative single-patient estimate")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=72)
            with col2:
                bp = st.number_input(
                    "Blood pressure (bp)", min_value=0, max_value=300, value=160
                )
            with col3:
                heart_rate = st.number_input(
                    "Heart rate", min_value=0, max_value=250, value=105
                )
            with col4:
                temperature = st.number_input(
                    "Temperature (°C)",
                    min_value=25.0,
                    max_value=45.0,
                    value=38.5,
                    step=0.1,
                )

            patient = pd.DataFrame(
                [[age, bp, heart_rate, temperature]], columns=FEATURES
            )
            result = "High Risk" if model.predict(patient)[0] == 1 else "Low Risk"
            st.metric("Illustrative model output", result)

            csv_data = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download analyzed CSV",
                data=csv_data,
                file_name="Patient_Risk_Model_Output.csv",
                mime="text/csv",
            )
