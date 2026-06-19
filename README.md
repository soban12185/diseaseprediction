#  Disease Prediction System

An AI-powered disease prediction application built using **Python**, **Scikit-learn**, and **Streamlit**. The system predicts possible diseases based on user-reported symptoms using machine learning models and provides an interactive web interface for real-time predictions.

---

# Features

*  Predicts diseases based on user-selected symptoms.
*  Machine Learning-based disease classification.
*  Compares multiple classification algorithms.
*  Data preprocessing and feature engineering pipeline.
*  Interactive Streamlit web application.
*  Real-time disease prediction with a simple user interface.

---

# Tech Stack

| Category           | Technologies             |
| ------------------ | ------------------------ |
| Language           | Python                   |
| Machine Learning   | Scikit-learn             |
| Data Processing    | Pandas, NumPy            |
| Data Visualization | Matplotlib               |
| Frontend           | Streamlit                |
| Dataset            | Disease Symptoms Dataset |

---

# System Architecture

```text
               User
                 │
                 ▼
        Select Symptoms
                 │
                 ▼
      Data Preprocessing
                 │
                 ▼
     Feature Vector Creation
                 │
                 ▼
 Machine Learning Classifier
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
 Decision Tree Random Forest Naive Bayes
                 │
                 ▼
      Predicted Disease
                 │
                 ▼
      Display Prediction
```

---

#  Project Structure

```text
Disease-Prediction-System/
│
├── app.py                 # Streamlit application
├── model.py               # Machine learning model
├── preprocess.py          # Data preprocessing
├── train_model.py         # Model training
├── dataset/
│   ├── Training.csv
│   └── Testing.csv
├── models/
│   └── disease_model.pkl
├── requirements.txt
└── README.md
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/soban12185/diseaseprediction.git
cd diseaseprediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

---

# Machine Learning Workflow

1. Load the disease dataset.
2. Clean and preprocess symptom data.
3. Convert symptoms into machine-readable feature vectors.
4. Train multiple classification models.
5. Compare model performance.
6. Predict diseases based on user symptoms.
7. Display results through the Streamlit interface.

---

#  Models Used

* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes Classifier

---

#  Key Capabilities

* Disease Prediction
* Symptom-Based Classification
* Data Preprocessing
* Feature Engineering
* Model Comparison
* Real-Time Prediction
* Interactive Streamlit Dashboard

---

#  Screenshots

Add screenshots of:

* Home Page
* Symptom Selection
* Prediction Result
* Model Comparison (if available)

---

#  Future Improvements

* Deep Learning-based prediction models
* Confidence score for predictions
* Personalized health recommendations
* Multi-language support
* Doctor and hospital recommendation module
* Integration with healthcare APIs

---

#  Author

**Soban S**

AI Engineer | Generative AI Engineer | Python Developer

📧 [sobansoban12185@gmail.com](mailto:sobansoban12185@gmail.com)

🔗 GitHub: https://github.com/soban12185

🔗 LinkedIn: https://linkedin.com/in/soban-s-884759297

---

⭐ If you found this project useful, consider giving it a star.
