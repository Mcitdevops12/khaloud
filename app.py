import numpy as np
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")


@st.cache_data
def load_model_and_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = (diabetes.target > diabetes.target.mean()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return diabetes, model, accuracy


diabetes, model, accuracy = load_model_and_data()
feature_names = diabetes.feature_names

st.title("🩺 Simple Diabetes Prediction App")
st.write(
    "This demo uses a small scikit-learn model trained on the diabetes dataset to estimate whether a person is at higher risk of diabetes."
)
st.metric("Model accuracy", f"{accuracy * 100:.1f}%")

st.sidebar.header("Patient details")

# Create a default set of values based on the data distribution.
default_values = [float(np.mean(diabetes.data[:, i])) for i in range(diabetes.data.shape[1])]
feature_ranges = [
    (float(np.min(diabetes.data[:, i])), float(np.max(diabetes.data[:, i])))
    for i in range(diabetes.data.shape[1])
]

inputs = {}
for name, (low, high), default in zip(feature_names, feature_ranges, default_values):
    value = st.sidebar.slider(name, low, high, round(default, 2))
    inputs[name] = value

if st.sidebar.button("Use sample data"):
    sample_values = {
        "age": 0.05,
        "sex": 0.03,
        "bmi": 0.37,
        "bp": 0.15,
        "s1": 0.06,
        "s2": 0.05,
        "s3": 0.04,
        "s4": 0.03,
        "s5": 0.09,
        "s6": 0.12,
    }
    for key, value in sample_values.items():
        inputs[key] = value

input_array = np.array([[inputs[name] for name in feature_names]], dtype=float)

prediction = model.predict(input_array)[0]
probability = model.predict_proba(input_array)[0][1]

if prediction == 1:
    result_text = "High risk"
    result_color = "🔴"
else:
    result_text = "Low risk"
    result_color = "🟢"

st.subheader("Prediction")
st.write(f"{result_color} Estimated risk: {result_text}")
st.write(f"Probability of high risk: {probability * 100:.1f}%")

st.caption("Tip: Adjust the sliders or click 'Use sample data' to try different inputs.")
