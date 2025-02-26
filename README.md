
# ANN-Classification-Churn-Modelling



A brief description of what this project does and who it's for

📊 Customer Churn Prediction

This project is a web application that predicts customer churn using an Artificial Neural Network (ANN) model. The model is trained on customer data and deployed using Streamlit for real-time predictions.

🚀 Live Demo

You can access the deployed application here:👉 https://ann-classification-churn-modelling-g6599.streamlit.app/

📌 Project Overview

Customer churn refers to the rate at which customers stop doing business with a company. This project aims to predict customer churn based on various input features such as credit score, age, account balance, tenure, and activity status.

🔹 How It Works

Users enter customer details through a web interface.

The input data is preprocessed using StandardScaler, LabelEncoder, and OneHotEncoder.

The trained ANN model makes predictions and displays the probability of churn.

💂️ Project Structure

📁 Customer-Churn-Prediction
🌀-- app.py                   # Streamlit application
🌀-- experiments.ipynb         # Jupyter Notebook with model training and evaluation
🌀-- model.h5                  # Trained ANN model
🌀-- scaler.pkl                # StandardScaler for feature scaling
🌀-- label_encoder_gender.pkl  # Label encoder for gender
🌀-- onehot_encoder_geo.pkl    # OneHotEncoder for geography
🌀-- requirements.txt          # Dependencies
🌀-- README.md                 # Project documentation

🛠️ Tech Stack & Tools

Category

Tools & Libraries

Programming

Python

Web Framework

Streamlit

Machine Learning

TensorFlow, Keras, Scikit-learn

Data Processing

Pandas, NumPy

Visualization

Matplotlib, Seaborn

🔧 Installation & Setup

Follow these steps to run the project locally:

1️⃣ Clone the Repository

git clone https://github.com/goutham6056/ANN-Classification-Churn-Modelling.git cd your-repo-name

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py

🧐 Machine Learning Model Details

Algorithm: Artificial Neural Network (ANN)

Framework: TensorFlow/Keras

Activation Functions:

ReLU (Hidden layers)

Sigmoid (Output layer)

Loss Function: Binary Crossentropy

Optimizer: Adam

📊 Input Features:

Feature

Description

Credit Score

Numeric value representing the customer’s creditworthiness

Gender

Male/Female (Label Encoded)

Age

Customer's age

Tenure

Number of years with the bank

Balance

Account balance

Number of Products

Number of bank products owned

Has Credit Card

Whether the customer has a credit card (Yes/No)

Is Active Member

Whether the customer is active (Yes/No)

Estimated Salary

Customer's salary

Geography

Country (One-Hot Encoded)

🎯 Features

👉 User-friendly web interface powered by Streamlit👉 Real-time predictions for customer churn👉 Deep learning model for accurate classification👉 Preprocessed dataset with encoding and scaling

📝 License

This project is licensed under the MIT License.

💡 Contributions are welcome! Feel free to fork this repo, raise issues, or submit pull requests.

🚀 Developed with ❤️ using Python, TensorFlow, and Streamlit!
