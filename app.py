# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/saiva/Downloads/ss/Crop_recommendation.csv")


# Train model
@st.cache_resource
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    # Better train-test split (stratified for balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Improved Random Forest (better tuning)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, X.columns


# Load and train
df = load_data()
model, accuracy, feature_names = train_model(df)


# Title
st.title("Crop Recommendation System")
st.write("Predict suitable crops based on soil nutrients and weather conditions")


# Show accuracy
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy*100:.2f}%")


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

N = st.number_input("Nitrogen (N) (kg/ha)", min_value=0.0)
P = st.number_input("Phosphorus (P) (kg/ha)", min_value=0.0)
K = st.number_input("Potassium (K) (kg/ha)", min_value=0.0)
temperature = st.number_input("Temperature (C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall (mm)")


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