import pickle
import numpy as np
import streamlit as st

# Load the model and scalar
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Title of the web app
st.title("Boston House Price Prediction")

# Input fields
CRIM = st.number_input("Enter CRIM", min_value=0.0)
ZN = st.number_input("Enter ZN", min_value=0.0)
INDUS = st.number_input("Enter INDUS", min_value=0.0)
CHAS = st.number_input("Enter CHAS (0 or 1)", min_value=0, max_value=1)
NOX = st.number_input("Enter NOX", min_value=0.0)
RM = st.number_input("Enter RM", min_value=0.0)
Age = st.number_input("Enter Age", min_value=0.0)
DIS = st.number_input("Enter DIS", min_value=0.0)
RAD = st.number_input("Enter RAD", min_value=0)
TAX = st.number_input("Enter TAX", min_value=0.0)
PTRATIO = st.number_input("Enter PTRATIO", min_value=0.0)
B = st.number_input("Enter B", min_value=0.0)
LSTAT = st.number_input("Enter LSTAT", min_value=0.0)

# When the 'Predict' button is pressed
if st.button('Predict'):
    # Prepare the data
    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, Age, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    scaled_data = scalar.transform(input_data)
    
    # Make prediction
    prediction = regmodel.predict(scaled_data)[0]
    
    # Display the prediction
    st.success(f"The predicted house price is: ${prediction:.2f}")