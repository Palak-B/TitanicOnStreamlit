import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained RandomForest model
model = joblib.load('titanic_model.pkl')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Streamlit UI
st.title("Titanic Survival Prediction")
st.write("Please enter the passenger details:")

# Get user inputs
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0, value=50)
pclass = st.selectbox("Pclass", options=[1, 2, 3], index=0)
sex = st.selectbox("Sex", options=["male", "female"], index=0)
embarked = st.selectbox("Embarked", options=["C", "Q", "S"], index=2)
title = st.selectbox("Title", options=["Mr", "Miss", "Mrs", "Master"], index=0)
cabin = st.text_input("Cabin", value="C85")  # Example of cabin input

# Age Bucket and Fare Bucket
age_bucket = st.selectbox("Age Bucket", options=["0-18", "19-35", "36-50", "51+"], index=0)
fare_bucket = st.selectbox("Fare Bucket", options=["Low", "Medium", "High"], index=0)

# Button to trigger prediction
if st.button("Predict Survival"):
    # Prepare input data as a DataFrame, mimicking the training data format
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [0],  # Assuming 'SibSp' is 0
        'Parch': [0],  # Assuming 'Parch' is 0
        'Fare': [fare],
        'Embarked': [embarked],
        'Sex': [sex],
        'FamilySize': [0],  # Assuming FamilySize is 0 for simplicity
        'Title': [title],
        'Cabin': [cabin],
        'Ticket': ['12345'],  # Example placeholder for Ticket
        'IsAlone': [1],  # Assuming the passenger is alone by default
        'Deck': ['C'],  # Example placeholder for Deck
        'AgeBucket': [age_bucket],
        'FareBucket': [fare_bucket],
        'HasCabin': [1],  # Assuming the passenger has a cabin
        'HasFamily': [1]  # Assuming the passenger has family
    })

    # Apply label encoding for categorical columns (Sex, Embarked, Title)
    input_data['Sex'] = label_encoder.fit_transform(input_data['Sex'])
    input_data['Embarked'] = label_encoder.fit_transform(input_data['Embarked'])
    input_data['Title'] = label_encoder.fit_transform(input_data['Title'])
    
    # Manually encode Cabin and Deck if needed (assuming it was one-hot encoded or used directly)
    input_data['Cabin'] = label_encoder.fit_transform(input_data['Cabin'])
    input_data['Deck'] = label_encoder.fit_transform(input_data['Deck'])

    # Encode AgeBucket and FareBucket as numbers
    age_bucket_map = {"0-18": 0, "19-35": 1, "36-50": 2, "51+": 3}
    fare_bucket_map = {"Low": 0, "Medium": 1, "High": 2}

    input_data['AgeBucket'] = input_data['AgeBucket'].map(age_bucket_map)
    input_data['FareBucket'] = input_data['FareBucket'].map(fare_bucket_map)

    # Predict using the model
    prediction = model.predict(input_data)
    
    # Show result
    if prediction == 1:
        st.write("The passenger survived.")
    else:
        st.write("The passenger did not survive.")
