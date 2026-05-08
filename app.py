# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")


# Train model
@st.cache_resource
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Tuned Random Forest Model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return model, acc, precision, recall, f1, X.columns


# Load and train
df = load_data()
model, accuracy, precision, recall, f1, feature_names = train_model(df)


# Title
st.title("Crop Recommendation System")
st.write("Predict suitable crops based on soil nutrients and weather conditions")


# Show performance
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy*100:.2f}%")
st.write(f"Precision: {precision*100:.2f}%")
st.write(f"Recall: {recall*100:.2f}%")
st.write(f"F1 Score: {f1*100:.2f}%")


# Feature Importance Graph
st.subheader("Feature Importance")

importances = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(feature_names, importances)
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
st.pyplot(fig)


# Input Section
st.subheader("Enter Input Values")

N = st.number_input("Nitrogen (N) (kg/ha)", min_value=0.0, max_value=140.0)
P = st.number_input("Phosphorus (P) (kg/ha)", min_value=0.0, max_value=145.0)
K = st.number_input("Potassium (K) (kg/ha)", min_value=0.0, max_value=205.0)

temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0)


# Prediction
if st.button("Predict Crop"):

    if N == 0 or P == 0 or K == 0:
        st.warning("Please enter valid values for all inputs")
    else:
        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=feature_names
        )

        prediction = model.predict(input_data)

        st.success(f"Recommended Crop: {prediction[0]}")
