import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the preprocessor and model
def load_objects():
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
    model_path = os.path.join('artifacts', 'model.pkl')
    
    with open(preprocessor_path, 'rb') as file:
        preprocessor = pickle.load(file)
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    return preprocessor, model

preprocessor, model = load_objects()

# Streamlit app
st.title('Diamond Price Prediction')

# Input fields
st.header('Enter Diamond Characteristics:')

carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color Rating:', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.number_input('Depth Percentage:', min_value=0.1, max_value=100.0, value=60.0)
table = st.number_input('Table Percentage:', min_value=0.1, max_value=100.0, value=55.0)
x = st.number_input('Length (X) in mm:', min_value=0.1, max_value=100.0, value=5.0)
y = st.number_input('Width (Y) in mm:', min_value=0.1, max_value=100.0, value=5.0)
z = st.number_input('Depth (Z) in mm:', min_value=0.1, max_value=100.0, value=3.0)

# Create a dataframe from inputs
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
})

# Predict button
if st.button('Predict Price'):
    # Display raw input data
    st.subheader("Raw Input Data:")
    st.write(input_data)

    # Preprocess the input
    input_processed = preprocessor.transform(input_data)
    
    # Display processed input
    st.subheader("Processed Input Data:")
    st.write(pd.DataFrame(input_processed, columns=preprocessor.get_feature_names_out()))

    # Make prediction
    prediction = model.predict(input_processed)
    
    # Display the result
    st.success(f'The predicted price of the diamond is ${prediction[0]:,.2f}')

    # Display model coefficients (if it's a linear model)
    if hasattr(model, 'coef_'):
        st.subheader("Model Coefficients:")
        coef_df = pd.DataFrame({
            'Feature': preprocessor.get_feature_names_out(),
            'Coefficient': model.coef_
        })
        st.write(coef_df)

# Optionally, add some information about the model and features
st.sidebar.header('About')
st.sidebar.info('This app predicts the price of a diamond based on its characteristics using a machine learning model.')
st.sidebar.header('Features Used')
st.sidebar.markdown("""
- Carat Weight: Weight of the diamond
- Cut: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- Color: Diamond color, from J (worst) to D (best)
- Clarity: A measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- Depth: Total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- Table: Width of top of diamond relative to widest point (43--95)
- X: Length in mm (0--10.74)
- Y: Width in mm (0--58.9)
- Z: Depth in mm (0--31.8)
""")

# Display preprocessor and model information
st.sidebar.header('Model Information')
st.sidebar.text(f"Preprocessor type: {type(preprocessor).__name__}")
st.sidebar.text(f"Model type: {type(model).__name__}")