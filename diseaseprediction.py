# --------------------------------------------
# AI Medical Disease Prediction Chatbot 💊
# --------------------------------------------

# 📦 Install required libraries before running:
# pip install streamlit scikit-learn pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# 🧠 Load and Prepare Datasetgit remote add origin https
# -------------------------------
# Replace 'disease_dataset.csv' with your actual file
df = pd.read_csv("Training.csv")

# Last column is 'prognosis' (disease name)
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 💬 Define Advice for Some Diseases
# -------------------------------
advice_dict = {
    "Allergy": "Avoid allergens and take antihistamines if needed. Stay hydrated.",
    "Fungal infection": "Use antifungal creams and keep affected areas clean and dry.",
    "Drug Reaction": "Consult your doctor immediately and avoid self-medication.",
    "Malaria": "Get tested quickly and take prescribed anti-malarial drugs.",
    "Dengue": "Drink plenty of fluids and monitor fever. See a doctor if symptoms worsen.",
    "Typhoid": "Maintain hygiene, drink boiled water, and follow your doctor’s antibiotics.",
    "Common Cold": "Rest well, drink fluids, and take steam inhalation.",
    "COVID-19": "Isolate yourself, get tested, and follow medical guidelines.",
}

# -------------------------------
# ⚙️ Prediction Function
# -------------------------------
def predict_disease(symptoms_list, top_n=3):
    # Create a vector of 0s
    input_data = [0] * len(X.columns)

    # Mark symptoms present
    for symptom in symptoms_list:
        if symptom in X.columns:
            input_data[X.columns.get_loc(symptom)] = 1

    # Get probabilities
    probabilities = model.predict_proba([input_data])[0]

    # Sort and get top N diseases
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_diseases = [(le.inverse_transform([i])[0], probabilities[i] * 100) for i in top_indices]

    return top_diseases

# -------------------------------
# 🌐 Streamlit UI
# -------------------------------
st.title("🩺 AI Medical Disease Prediction Chatbot")
st.write("Enter your symptoms below to get possible diseases and suggestions.")

# Multiple symptom input (from list)
symptom_options = list(X.columns)
selected_symptoms = st.multiselect("Select your symptoms:", options=symptom_options)

# Predict button
if st.button("🔍 Predict Disease"):
    if selected_symptoms:
        predictions = predict_disease(selected_symptoms, top_n=3)

        st.subheader("🧠 Predicted Probabilities:")
        for disease, prob in predictions:
            st.write(f"- **{disease}**: {prob:.2f}%")

        top_disease = predictions[0][0]
        st.success(f"💊 Most likely disease: **{top_disease}**")

        if top_disease in advice_dict:
            st.info(f"🩹 Suggested Advice: {advice_dict[top_disease]}")
        else:
            st.info("🩹 Suggested Advice: Please consult a healthcare professional for accurate treatment.")
    else:
        st.warning("Please select at least one symptom.")

# -------------------------------
# 📊 Optional: Visual Probability Bar
# # -------------------------------
#     prob_df = pd.DataFrame(predictions, columns=["Disease", "Probability"])
#     st.bar_chart(prob_df.set_index("Disease"))
st.markdown("""
### 🩺 **Disclaimer**
> This application uses AI-based predictions for informational purposes only.  
> It is **not a substitute for professional medical advice, diagnosis, or treatment.**  
> Always consult a qualified **healthcare professional** before making medical decisions.
""")
