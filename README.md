# 🚦 UrbanFlow Green Mobility Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)

An enterprise-grade smart-city traffic intelligence platform and decision-support web application. **UrbanFlow Green Mobility Intelligence** leverages multi-target XGBoost pipelines to deliver real-time congestion classification, intersection wait time regression, and carbon emission modeling, featuring local explainability and printable executive reporting.

---

## 🗝️ Key Features

* **Multi-Target Machine Learning Pipeline:**
  * **Congestion Classification:** Categorizes traffic density into actionable status levels.
  * **Wait Time Estimation:** Predicts intersection queue wait times in minutes using log-transformed XGBoost regression.
  * **Carbon Emission Tracking:** Calculates localized vehicular $\text{CO}_2$ output ($\text{kg}$) to support municipal green transit goals.
* **Dual Operating Modes:**
  * **Interactive Single Node:** Modify 38+ real-time environmental, temporal, and IoT sensor metrics using intuitive UI sliders and expanders.
  * **Bulk CSV Batch Evaluation:** Process full traffic datasets simultaneously and download enriched CSV prediction outputs.
* **Model Explainability & Insights:**
  * Displays pre-aligned global feature importances extracted via scikit-learn preprocessing pipelines (`feature_importances.joblib`).
* **Executive PDF/HTML Reporting:**
  * Generates clean, browser-printable Executive Brief reports summarizing node parameters and predictive outputs.
* **Production-Grade MLOps Architecture:**
  * Strict schema enforcement and robust fallback handling using serialized `.joblib` artifacts.

---

## 📁 Repository Structure

```text
├── app.py                            # Streamlit web application dashboard
├── ufgmi_mlp.ipynb                   # End-to-end ML training & experimentation notebook
├── requirements.txt                  # Version-locked production dependencies
├── standard_preprocessor.joblib      # Fitted scikit-learn preprocessing pipeline
├── feature_importances.joblib        # Pre-calculated 56-feature importance dataframe
├── categorical_mappings.joblib       # Label encoding and categorical options metadata
├── numerical_metadata.joblib         # Continuous feature boundaries and default values
├── model_xgb_congestion_classifier.joblib # Trained XGBoost classification model (Congestion Level)
├── model_xgb_wait_time.joblib        # Trained XGBoost regression model (Wait Time in Minutes)
├── model_xgb_emissions.joblib        # Trained XGBoost regression model (CO2 Emissions in Kilograms)
├── congestion_label_encoder.joblib   # LabelEncoder artifact for target classes
└── README.md                         # Project documentation
```

---

## 💾 Quickstart (Local Installation)

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/ayman-nader/UrbanFlow-Green-Mobility-Intelligence-ML-Project
cd UrbanFlow-Green-Mobility-Intelligence
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🗄️ Dataset Attribution & Licensing

* **Dataset Title:** CityFlow - Smart Urban Mobility & Traffic IoT
* **Source:** [Kaggle Dataset Datacard](https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot)
* **Author:** Mobeen Fatimah
* **Description:** Contains high-dimensional IoT sensor data, roadway configurations, weather metrics, and traffic density parameters across urban intersection nodes.
* **License:** Shared under Kaggle Platform standard data usage policies for research, educational, and portfolio development.

---

## 🛠️ Built With

* **Frontend & Web Framework:** [Streamlit](https://streamlit.io/)
* **Machine Learning Engine:** [XGBoost](https://xgboost.readthedocs.io/) & [scikit-learn](https://scikit-learn.org/)
* **Data Processing:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Interactive Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
* **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)

---

## 📄 License & Usage
This project is developed for educational, analytical, and professional portfolio demonstration purposes, utilizing the [CityFlow Kaggle Dataset](https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot).