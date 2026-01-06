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
# -------------------------------
# 🎨 Custom CSS & Styling
# -------------------------------
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #f0f8ff; /* Light Alice Blue */
            color: #333;
        }

        /* App Background */
        .stApp {
            background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
        }

        /* Title Style */
        .title-container {
            text-align: center;
            background-color: #00796b; /* Teal */
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        .title-container h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
        }
        .title-container p {
            font-size: 1.1rem;
            margin-top: 10px;
            opacity: 0.9;
        }

        /* Card Style for Inputs */
        .input-card {
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #00796b;
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 121, 107, 0.3);
        }
        div.stButton > button:hover {
            background-color: #004d40;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 121, 107, 0.4);
            color: white;
        }

        /* Success Message Card */
        .result-card {
            background: linear-gradient(to right, #00b09b, #96c93d);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .result-card h2 {
            color: white;
            margin: 0;
        }

        /* Advice Box */
        .advice-box {
            background-color: #fff3e0;
            border-left: 6px solid #ff9800;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            color: #e65100;
            font-size: 1.05rem;
        }

        /* Disclaimer */
        .disclaimer {
            font-size: 0.85rem;
            color: #777;
            text-align: center;
            margin-top: 50px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🌐 Streamlit UI
# -------------------------------

# Custom HTML Title
st.markdown("""
    <div class="title-container">
        <h1>🩺 AI Health Guard</h1>
        <p>Your Intelligent Disease Prediction Assistant</p>
    </div>
""", unsafe_allow_html=True)

# Input Section wrapped in a container explicitly if needed, but here we just use the flow
# To mimic a card, we can't easily wrap Streamlit widgets in custom HTML divs without components,
# but we can style the surrounding elements.
# Alternatively, just let the global CSS handle the 'white' look if possible, or use columns.

st.write("### 📝 Describe Your Symptoms")
st.markdown('<div class="input-card">', unsafe_allow_html=True)

# Multiple symptom input (from list)
symptom_options = list(X.columns)
selected_symptoms = st.multiselect("Select your symptoms from the list below:", options=symptom_options)

st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("🔍 Analyze Symptoms"):
    if selected_symptoms:
        with st.spinner('Processing your symptoms...'):
            predictions = predict_disease(selected_symptoms, top_n=3)
            
            # Top Prediction Result
            top_disease = predictions[0][0]
            top_prob = predictions[0][1]

            st.markdown(f"""
                <div class="result-card">
                    <h2>Diagnosis: <b>{top_disease}</b></h2>
                    <p>Confidence Level: {top_prob:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # Advice
            if top_disease in advice_dict:
                st.markdown(f"""
                    <div class="advice-box">
                        <b>💡 Recommended Action:</b><br>
                        {advice_dict[top_disease]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown(f"""
                    <div class="advice-box">
                        <b>💡 Recommended Action:</b><br>
                        Please consult a healthcare professional for specific advice.
                    </div>
                """, unsafe_allow_html=True)

            # Detailed Breakdown Section
            st.markdown("### 📊 Detailed Analysis")
            prob_df = pd.DataFrame(predictions, columns=["Potential Disease", "Probability (%)"])
            prob_df.set_index("Potential Disease", inplace=True)
            st.table(prob_df)

    else:
        st.warning("⚠️ Please select at least one symptom to proceed.")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <p><b>⚠️ Medical Disclaimer:</b> This tool uses AI for informational purposes only and is not a substitute for professional medical diagnosis. Always consult a doctor for serious concerns.</p>
</div>
""", unsafe_allow_html=True)
