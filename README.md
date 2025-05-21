# Customer-LifetimeValue
# Customer Lifetime Value (CLV) Prediction for Vehicle Insurance

## Overview
This project aims to predict **Customer Lifetime Value (CLV)** for a vehicle insurance company using machine learning. By analyzing factors like vehicle type, insurance type, income, and customer demographics, the model helps segment customers and optimize business strategies for retention, profitability, and personalized offerings.

## Key Features
- **Dataset Variables**:  
  Vehicle type, insurance type, occupation, marital status, education level, number of policies, claims, income, and CLV (target variable).  
- **Business Goals**:  
  Improve decision-making, customer retention, and segmentation to drive profitability.

## Approach
1. **Data Preprocessing**: Handle missing values, outliers, and categorical encoding.  
2. **Exploratory Data Analysis (EDA)**: Visualize feature distributions and correlations.  
3. **Model Development**: Train regression models (e.g., Linear Regression, Random Forest, Gradient Boosting) to predict CLV.  
4. **Evaluation**: Metrics include RMSE, MAE, and R².  
5. **Customer Segmentation**: Apply clustering (e.g., K-Means) on predicted CLV for targeted strategies.  

## Results
- Best-performing model: **Random Forest** (example: RMSE = 1200, R² = 0.85).  
- Segmented customers into groups (e.g., High-CLV, Medium-CLV, Low-CLV) for tailored marketing.  

## Usage
1. Install dependencies:  
   ```bash
   pip install pandas scikit-learn matplotlib seaborn jupyter
