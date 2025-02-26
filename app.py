import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import numpy as np
import pandas as pd
import streamlit as st

model = tf.keras.models.load_model('model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder_gender.pkl', 'rb'))
one_hot_encoder = pickle.load(open('onehot_encoder_geo.pkl', 'rb'))

# streamlit app
st.title("Customer Churn Prediction")
st.subheader("Enter the customer details below")

# Input fields for customer details

geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.number_input('Age', min_value=18, max_value=100,
                      value=30)  # default age is 30
# default balance is 10000.0
balance = st.number_input('Balance', min_value=0.0, value=10000.0)
credit_score = st.number_input('Credit Score')  # default credit score is 650
# default estimated salary is 50000.0
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10, 5)  # default tenure is 5 years
# default number of products is 2
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
if has_cr_card == 'Yes':
    has_cr_card = 1
else:
    has_cr_card = 0
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
if is_active_member == 'Yes':
    is_active_member = 1
else:
    is_active_member = 0

# prepare the input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geography_encoded = one_hot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(
    geography_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(
    drop=True), geography_encoded_df], axis=1)

# sca;le the input data
input_data_scaled = scaler.transform(input_data)

# make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")
st.write(f"Prediction Probability: {prediction_proba:.2f}")
