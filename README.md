#  Olist Delivery Time Prediction Dashboard

An end-to-end Machine Learning project that predicts **e-commerce delivery time (in days)** and explains the predictions using **feature importance, SHAP analysis, KPI metrics, and business insights**, deployed as an interactive **Streamlit dashboard**.

---

# 🌐 Live Demo
https://olist-delivery-time-prediction-menqktrznudvjuutmmupt7.streamlit.app/
---
##  Project Overview

Timely delivery is critical for e-commerce platforms.  
This project predicts how long an order will take to be delivered using historical order, seller, and logistics data from the **Olist Brazilian E-commerce Dataset**.

The project includes:
- A trained ML regression model
- A production-ready Streamlit dashboard
- Explainable AI (SHAP)
- KPI analytics
- Business optimization recommendations
- Cloud deployment on Streamlit

---

##  Problem Statement

Predict the delivery time (in days) for an e-commerce order at the time of purchase.

This helps:
- Customers estimate delivery expectations
- Platforms optimize logistics
- Businesses identify delay bottlenecks

---

##  Dataset

Source:  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Files used:
- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_sellers_dataset.csv

Key information:
- Order timestamps
- Delivery timestamps
- Freight value (shipping cost)
- Seller history
- Order value and payment amount
- Temporal features (hour, day, month)

---

##  Feature Engineering

Key engineered features:
- Delivery time (target) = Delivered date − Purchase date
- Purchase hour, day of week, month
- Seller order count (experience proxy)
- Total price, total freight, payment total
- Number of items

Product category features were tested and later removed due to low predictive impact.

---

##  Model Details

- Model Type: XGBoost Regressor
- Task: Regression
- Target: Delivery time (days)

Model performance:
- MAE ≈ 5.15 days
- RMSE ≈ 8.31 days

---

##  Model Explainability

Feature Importance:
- Identifies globally important predictors

Top drivers:
- Total freight (strongest)
- Seller order count
- Payment total
- Purchase month
- Total price

SHAP Analysis:
- Explains why predictions increase or decrease
- High freight → longer delivery
- Experienced sellers → faster delivery
- Seasonal effects are clearly visible

---

##  What Does “Freight” Mean?

Freight refers to the shipping cost charged to deliver an order.

It acts as a proxy for:
- Distance between seller and customer
- Logistics complexity
- Handling effort

Higher freight usually indicates longer delivery time.

---

##  Streamlit Dashboard

The dashboard is organized into four tabs:

1. Prediction  
   - User inputs order details  
   - Model predicts delivery time  

2. Findings & Analysis  
   - KPI summary cards  
   - Feature importance plot  
   - SHAP summary plot  

3. Optimization Strategies  
   - Actionable delivery improvement recommendations  

4. Model Interpretation (Q&A)  
   - Explains model behavior  
   - Seasonal and behavioral insights  
   - Practical conclusions  

---

##  Project Structure

.
├── app.py  
├── delivery_time_model.pkl  
├── model_columns.json  
├── Feature_imp_olist.png  
├── shap_olist.png  
├── olist_orders_dataset.csv  
├── olist_order_items_dataset.csv  
├── olist_sellers_dataset.csv  
├── requirements.txt  
└── README.md  

---

##  Key Learnings

- ML models require training-time dependencies at inference
- Freight is the strongest proxy for delivery delay
- Seller experience improves delivery reliability
- Explainability is critical for trust in ML systems
- Linux deployment is case-sensitive (file names matter)

---

##  Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Joblib
- Streamlit

---


##  Acknowledgements

Dataset provided by Olist (Brazilian E-commerce).
