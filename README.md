# 🫀 Heart Disease Predictor

A clinical-grade **Streamlit web application** for heart disease risk prediction using a Logistic Regression ML pipeline. Enter patient details to instantly receive a personalised risk assessment backed by statistical feature analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Preview

> Upload your CSV → Enter patient details → Get instant prediction with risk score, patient report card, and model performance metrics.

---

## ✨ Features

- **Patient Report** — Enter name, age, cholesterol, blood pressure, and smoking status to generate a personalised risk report
- **Risk Score** — Visual probability bar showing percentage likelihood of heart disease
- **Result Cards** — Color-coded output (🔴 High Risk / 🟢 Healthy) with the patient's name prominently displayed
- **Model Performance Tab** — Accuracy, Precision, Recall, Confusion Matrix, and full Classification Report
- **Feature Analysis Tab** — T-Test (numerical) and Chi-Square (categorical) significance tests with p-values
- **Feature Distribution Explorer** — Compare value distributions between disease and non-disease groups
- **Dark Clinical UI** — Polished dark-themed interface built with custom CSS

---

## 🗂️ Project Structure

```
heart-disease-predictor/
│
├── heart_disease_app.py       # Main Streamlit application
├── heart_disease_50000.csv    # Dataset (not included — add your own)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📋 Dataset Requirements

Your CSV file must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `age` | Numerical | Patient age in years |
| `cholesterol` | Numerical | Cholesterol level (mg/dL) |
| `blood_pressure` | Numerical | Blood pressure reading (mmHg) |
| `smoking_status` | Categorical | Smoking habit category |
| `heart_disease` | Binary (0/1) | Target — 0 = No Disease, 1 = Disease |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run heart_disease_app.py
```

### 4. Open in browser

The app opens automatically at `http://localhost:8501`

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
```

Or install directly:

```bash
pip install streamlit pandas numpy scikit-learn scipy
```

---

## 🧠 ML Pipeline

```
CSV Upload
    │
    ▼
Data Cleaning (drop duplicates)
    │
    ▼
Statistical Feature Selection
    ├── T-Test  → Numerical features (age, cholesterol, blood_pressure)
    └── Chi-Square → Categorical features (smoking_status)
    │
    ▼
ColumnTransformer
    ├── StandardScaler   → age, cholesterol, blood_pressure
    └── OneHotEncoder    → smoking_status
    │
    ▼
Logistic Regression (max_iter=1000, random_state=42)
    │
    ▼
Evaluation → Accuracy · Precision · Recall · Confusion Matrix
```

---

## 🖥️ How to Use

1. **Upload** your `heart_disease_50000.csv` file using the sidebar uploader
2. **Enter Patient Details** in the sidebar:
   - Patient Name
   - Age (slider)
   - Cholesterol level (slider)
   - Blood Pressure (slider)
   - Smoking Status (dropdown — auto-populated from your data)
3. Click **🔍 Run Prediction**
4. View the **personalised patient report** with risk probability
5. Explore model metrics and feature analysis in the other tabs

---

## 📊 App Tabs

| Tab | Contents |
|---|---|
| 🔮 Prediction | Patient result card, risk probability bar, patient summary |
| 📊 Model Performance | Accuracy/Precision/Recall, Confusion Matrix, Classification Report, Dataset stats |
| 🔬 Feature Analysis | Significance test table, significant features list, distribution explorer |

---

## ⚠️ Disclaimer

> This application is built for **educational and demonstration purposes only**.
> It is **not a substitute for professional medical diagnosis or advice**.
> Always consult a qualified healthcare professional for medical decisions.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

Built with ❤️ using Python, Streamlit, and scikit-learn.

> *Star ⭐ this repo if you found it helpful!*
