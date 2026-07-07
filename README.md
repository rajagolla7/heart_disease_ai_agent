# ❤️ Heart Disease AI — Risk Prediction Dashboard

A premium, interactive **heart disease risk prediction** web app built with **Streamlit** and a trained **Random Forest** model. Enter patient clinical data and get an instant High Risk / Low Risk verdict with health recommendations.

---

## 🚀 Features

- **Live Single Prediction** — Fill in 13 clinical features and get an instant prediction
- **Batch CSV Predictions** — Upload a CSV file to predict for multiple patients at once
- **Feature Importance Chart** — Understand which features drive the model's decisions
- **Health Recommendations** — Personalized advice based on the risk level
- **CSV Export** — Download detailed prediction results

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | [Streamlit](https://streamlit.io/) |
| ML Model | Random Forest (scikit-learn) |
| Data | [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| Visualization | Plotly |
| Language | Python 3.x |

---

## 📦 Installation

### 1. Clone / Download the project

```bash
git clone https://github.com/your-username/heart-disease-ai.git
cd heart-disease-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
python -m streamlit run app.py
```

Then open your browser at: **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Dataset

The model is trained on the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) containing **1,025 patient records** with 13 clinical features:

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Sex (0 = Female, 1 = Male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (0 = No, 1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels (0–4) |
| `thal` | Thalassemia (0–3) |

**Target:**
- `0` → Heart disease detected (High Risk)
- `1` → No heart disease (Low Risk)

---

## 📁 Project Structure

```
heart_disease_ai_agent-main/
│
├── app.py                  # Main Streamlit application
├── heart_model .pkl        # Pre-trained Random Forest model
├── heart.csv               # Training dataset
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```

---

## ⚠️ Disclaimer

> This application is intended for **educational and demonstration purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 📄 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
