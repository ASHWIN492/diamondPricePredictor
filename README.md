# Diamond Price Prediction

This project is a web application that predicts the price of a diamond based on its characteristics using a machine learning model. The application is built using Python, with Streamlit as the web framework, and Scikit-learn for the machine learning components.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build a regression model to predict the price of diamonds. The model uses various features such as carat weight, cut, color, clarity, depth, table, and dimensions (x, y, z) of the diamond. The app allows users to input these features and predicts the price of the diamond in real-time.

## Features
- **Real-time Price Prediction:** Users can input diamond characteristics and get a price prediction instantly.
- **Preprocessing Pipeline:** The app uses a preprocessing pipeline that handles missing values, encoding categorical features, and scaling numerical features.
- **Model Training:** The application trains several machine learning models and selects the best-performing model based on R² score.
- **User-Friendly Interface:** The app is built using Streamlit, making it easy to use and interact with.

## Installation

### Prerequisites
- Python 3.x
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diamond-price-prediction.git
2. Navigate to the project directory
   cd diamond-price-prediction
3. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
4. Install the required packages:
   pip install -r requirements.txt
5. Run the Streamlit app
   Run the Streamlit app

## File Structure
    ├── app.py                 # The main Streamlit app file
    ├── src/
    │   ├── components/
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   ├── model_trainer.py
    │   ├── logger.py
    │   ├── exception.py
    │   ├── utils.py
    ├── artifacts/             # Directory for storing artifacts (e.g., models, preprocessors)
    ├── notebooks/             # Jupyter notebooks for experimentation
    ├── training_pipeline.py   # Script for training the machine learning model
    ├── prediction_pipeline.py # Script for making predictions with the trained model
    ├── requirements.txt       # List of required Python packages
    └── README.md              # This file

## Model Information
The model is trained on a dataset of diamonds with various features. It uses the following machine learning models:

1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. ElasticNet Regression
5. Decision Tree Regressor    

## Model Training Output
During the model training process, the following output was observed:

model = cd_fast.enet_coordinate_descent(
{'LinearRegression': 0.8478189531683753, 'Lasso': 0.8869461803291463, 'Ridge': 0.8526480584829872, 'ElasticNet': 0.8245368382630797, 'DecisionTree': 0.9612189524948617}
Best Model Found, Model name: DecisionTree, R2 score: 0.9612189524948617


## Preprocessing Pipeline
The preprocessing pipeline includes:

1. Numerical Features: Imputation of missing values using the median and scaling using StandardScaler.
2. Categorical Features: Imputation of missing values using the most frequent strategy, ordinal encoding, and scaling.
