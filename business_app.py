import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Set page config
st.set_page_config(page_title="Sales Prediction App", layout="wide")

# Title and description
st.title("📊 Sales Prediction Model")
st.markdown("---")
st.write("""
## Business Context
A company wants to predict sales based on advertising budget.

**Goal:** Build a Linear Regression model that predicts sales using TV advertising budget.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Data Overview", "Model Training", "Predictions", "Model Performance"])

# Create sample data
data = {
    'TV_advertising': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'Sales': [5, 6, 7, 9, 10, 12, 13, 14, 15, 17]
}
df = pd.DataFrame(data)

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.Y_train = None
    st.session_state.Y_test = None
    st.session_state.y_pred = None
    st.session_state.metrics = None

# PAGE 1: Data Overview
if page == "Data Overview":
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset")
        st.dataframe(df, use_container_width=True)
        
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Data Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col3:
        st.metric("Features", len(df.columns) - 1)
    
    # Visualization
    st.subheader("Visualization: TV Advertising vs Sales")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['TV_advertising'], df['Sales'], s=100, alpha=0.6, color='blue')
    ax.set_xlabel("TV Advertising Budget", fontsize=12)
    ax.set_ylabel("Sales", fontsize=12)
    ax.set_title("Relationship between TV Advertising and Sales", fontsize=14)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# PAGE 2: Model Training
elif page == "Model Training":
    st.header("Model Training")
    
    if st.button("🚀 Train Model", key="train_button"):
        with st.spinner("Training model..."):
            # Prepare data
            X = df.drop("Sales", axis=1)
            Y = df["Sales"]
            
            # Split data
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=43
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, Y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.Y_train = Y_train
            st.session_state.Y_test = Y_test
            st.session_state.y_pred = y_pred
            
            st.success("✅ Model trained successfully!")
    
    if st.session_state.model is not None:
        st.info("✓ Model is ready for predictions")
        
        # Display model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Coefficient (TV_advertising)", f"{st.session_state.model.coef_[0]:.4f}")
        with col2:
            st.metric("Intercept", f"{st.session_state.model.intercept_:.4f}")

# PAGE 3: Predictions
elif page == "Predictions":
    st.header("Make Predictions")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first on the 'Model Training' page!")
    else:
        st.subheader("Predict Sales for New TV Advertising Budget")
        
        # Input for prediction
        tv_budget = st.slider(
            "Select TV Advertising Budget:",
            min_value=0,
            max_value=250,
            value=170,
            step=10
        )
        
        # Make prediction
        new_budget = np.array([[tv_budget]])
        predicted_sales = st.session_state.model.predict(new_budget)[0]
        
        # Display result
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("TV Advertising Budget", f"${tv_budget}")
        with col2:
            st.metric("Predicted Sales", f"${predicted_sales:.2f}", delta=None)
        
        st.markdown("---")
        
        # Manual input option
        st.subheader("Or Enter a Custom Value")
        custom_budget = st.number_input(
            "Enter TV Advertising Budget:",
            min_value=0,
            max_value=500,
            value=170
        )
        
        custom_prediction = st.session_state.model.predict(np.array([[custom_budget]]))[0]
        st.info(f"**Predicted Sales for ${custom_budget}: ${custom_prediction:.2f}**")

# PAGE 4: Model Performance
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first on the 'Model Training' page!")
    else:
        # Calculate metrics
        mae = mean_absolute_error(st.session_state.Y_test, st.session_state.y_pred)
        mse = mean_squared_error(st.session_state.Y_test, st.session_state.y_pred)
        r2 = r2_score(st.session_state.Y_test, st.session_state.y_pred)
        rmse = np.sqrt(mse)
        
        # Store metrics in session state
        st.session_state.metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        # Display metrics
        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.4f}")
        with col2:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col3:
            st.metric("RMSE", f"{rmse:.4f}")
        with col4:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Interpretation
        st.subheader("Metric Interpretation")
        st.write(f"""
        - **MAE (Mean Absolute Error):** {mae:.4f} - Average prediction error
        - **MSE (Mean Squared Error):** {mse:.4f} - Penalizes larger errors
        - **RMSE (Root Mean Squared Error):** {rmse:.4f} - In same units as target
        - **R² Score:** {r2:.4f} - Proportion of variance explained (1.0 is perfect)
        """)
        
        # Visualization: Actual vs Predicted
        st.subheader("Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(st.session_state.X_test, st.session_state.Y_test, label="Actual", s=100, alpha=0.6, color='blue')
        ax.plot(st.session_state.X_test, st.session_state.y_pred, label="Predicted", color='red', linewidth=2)
        ax.set_xlabel("TV Advertising Budget", fontsize=12)
        ax.set_ylabel("Sales", fontsize=12)
        ax.set_title("Linear Regression Model: Actual vs Predicted", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Residuals plot
        st.subheader("Residuals Analysis")
        residuals = st.session_state.Y_test - st.session_state.y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(st.session_state.y_pred, residuals, s=100, alpha=0.6, color='green')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel("Predicted Values", fontsize=12)
        ax.set_ylabel("Residuals", fontsize=12)
        ax.set_title("Residual Plot", fontsize=14)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📈 Sales Prediction Model | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
