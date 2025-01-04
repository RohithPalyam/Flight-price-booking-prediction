import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
st.title("Flight Booking Price Prediction")
data = pd.read_csv("Flight_Booking.csv")  # Replace with actual dataset path
global data_encoded

# Sidebar with option menu
with st.sidebar:
    selected = option_menu(
        "Tasks Menu",
        ["Data Inspection", "Data Visualization", "Feature Engineering"],
        icons=["clipboard-data", "bar-chart", "tools"],
        menu_icon="menu-app",
        default_index=0
    )

# Data Inspection
if selected == "Data Inspection":
    with st.echo():
        st.write("### Data Shape")
        st.write(data.shape)

        st.write("### Data Info")
        st.write(data.info())

        st.write("### Data Description")
        st.write(data.describe())

# Data Visualization
if selected == "Data Visualization":
    with st.echo():
        st.write("### Count of Flights by Airline")
        plt.figure(figsize=(8, 6))
        sns.countplot(x='airline', data=data)
        plt.title("Count of Flights by Airline")
        st.pyplot(plt)

        st.write("### Price Range by Class")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='class', y='price', data=data)
        plt.title("Price Range by Class")
        st.pyplot(plt)

        st.write("### Price vs Days Left by Source City")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='days_left', y='price', hue='source_city', data=data)
        plt.title("Price vs Days Left by Source City")
        st.pyplot(plt)

# Feature Engineering: One-Hot Encoding using pd.get_dummies()
if selected == "Feature Engineering":
    with st.echo():
        # Handle missing values by filling with mode for categorical columns
        data = data.fillna(data.mode().iloc[0])

        # Apply one-hot encoding using pandas get_dummies
        categorical_columns = ['airline', 'source_city', 'destination_city', 'class']
        data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        st.write("### Encoded Data")
        st.write(data_encoded.head())  # Check the encoded columns

        # Drop any non-numeric columns such as flight codes
        # Here, I'm assuming 'flight_code' or similar columns could be present, replace with actual column names if necessary
        data_encoded = data_encoded.select_dtypes(include=[np.number])  # Keep only numeric columns

        # Check if any columns remain non-numeric
        non_numeric_cols = data_encoded.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            st.write(f"Non-numeric columns detected: {non_numeric_cols}")
        else:
            st.write("All columns are numeric.")

        # Linear Regression Model
        X = data_encoded.drop('price', axis=1)  # Features
        y = data_encoded['price']  # Target variable

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)

        # Display Linear Regression metrics
        st.write("### Linear Regression Metrics")
        st.write("R2 Score:", r2_score(y_test, y_pred_lr))
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
