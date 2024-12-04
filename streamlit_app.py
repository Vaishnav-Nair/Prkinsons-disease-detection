import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
parkinsons_data = pd.read_csv('parkinsons.csv')

# Preprocessing
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train Models (SVM, Logistic Regression, Random Forest, KNN)
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, Y_train)

logreg_model = LogisticRegression()
logreg_model.fit(X_train, Y_train)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, Y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

# Streamlit app
st.title("Parkinson's Disease Prediction")

# Input for 22 features as a single line of comma-separated values
st.write("Paste the 22 features in a single line, comma-separated:")

# Text area to input comma-separated values
input_string = st.text_area('Enter comma-separated values for the 22 features:', '197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569')

# Split the input string and convert to float
try:
    input_data = [float(i) for i in input_string.split(',')]
    if len(input_data) != 22:
        st.error("Please enter exactly 22 values.")
except ValueError:
    st.error("Please enter valid numeric values separated by commas.")

# Model selection
model_choice = st.selectbox("Choose Model", ['SVM', 'Logistic Regression', 'Random Forest', 'KNN'])

# Button to predict
if st.button('Predict'):
    if len(input_data) == 22:
        input_data = np.asarray(input_data).reshape(1, -1)  # Reshape the input to match the expected input for prediction (1, 22)
        input_data = scaler.transform(input_data)  # Standardize the input

        # Predict using the selected model
        if model_choice == 'SVM':
            prediction = svm_model.predict(input_data)
        elif model_choice == 'Logistic Regression':
            prediction = logreg_model.predict(input_data)
        elif model_choice == 'Random Forest':
            prediction = rf_model.predict(input_data)
        elif model_choice == 'KNN':
            prediction = knn_model.predict(input_data)

        # Display the prediction result
        if prediction[0] == 0:
            st.success("The person is Healthy.")
        else:
            st.error("The person has Parkinson's Disease.")
