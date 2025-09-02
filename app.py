import streamlit as st
import pickle
import numpy as np

# Load the model, scaler, and encoder
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    encoder = data['encoder']

st.title('Iris Species Prediction')
st.write('Enter the features below to predict the Iris species:')

# Input fields for the four features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

if st.button('Predict'):
    # Prepare input
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)
    species = encoder.inverse_transform(prediction)[0]
    st.success(f'Predicted Iris Species: {species}')
