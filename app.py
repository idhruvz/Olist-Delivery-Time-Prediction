import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import traceback

st.set_page_config(page_title="Delivery Time Predictor", layout="wide")

# Load Model + Columns
@st.cache_resource
def load_model():
    model = joblib.load("delivery_time_model.pkl")
    with open("model_columns.json", "r") as f:
        columns = json.load(f)
    return model, columns

try:
    model, model_cols = load_model()
except Exception as e:
    st.error("❌ Model loading failed")
    st.exception(e)
    st.stop()


# Load Olist Data for KPI Analytics
@st.cache_data
def load_olist_data():
    orders = pd.read_csv("olist_orders_dataset.csv")
    items = pd.read_csv("olist_order_items_dataset.csv")
    sellers = pd.read_csv("olist_sellers_dataset.csv")

    # Convert timestamps
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])

    # Compute delivery time
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.days

    # Clean invalid delivery times
    orders = orders[(orders["delivery_days"] >= 0) & (orders["delivery_days"].notna())]

    # Add month column
    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.month

    # Merge freight value
    merged = orders.merge(items[["order_id", "freight_value"]], on="order_id", how="left")

    return merged

try:
    olist = load_olist_data()
except:
    st.error(" Could not load Olist CSV files. Check filenames.")
    st.stop()

# Compute KPI Values
avg_delivery = olist["delivery_days"].mean()
avg_freight = olist["freight_value"].mean()

# Monthly medians
median_by_month = olist.groupby("purchase_month")["delivery_days"].median()

fastest_month = median_by_month.idxmin()
slowest_month = median_by_month.idxmax()
most_orders_month = olist["purchase_month"].value_counts().idxmax()

# Month Name Mapping
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

fastest_month_name = month_names.get(fastest_month, "-")
slowest_month_name = month_names.get(slowest_month, "-")
most_orders_month_name = month_names.get(most_orders_month, "-")

# Header
st.title("Olist Delivery Time Predictor Dashboard")
st.write("Predict delivery times, view insights, and explore optimization strategies.")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Predict",
    "Findings & Analysis",
    "Optimization Strategies",
    "Q&A / Model Interpretation"
])

# TAB 1 — PREDICTION
with tab1:

    st.header("Enter Order Details")
    with st.form("predict_form"):

        col1, col2, col3 = st.columns(3)
        with col1:
            purchase_hour = st.number_input("Purchase Hour (0–23)", 0, 23, 12)
            purchase_dow = st.number_input("Day of Week (0=Mon)", 0, 6, 2)

        with col2:
            purchase_month = st.number_input("Month (1–12)", 1, 12, 5)
            num_items = st.number_input("Number of Items", 1, 50, 1)

        with col3:
            total_price = st.number_input("Total Price (BRL)", 0.0, 50000.0, 100.0)
            total_freight = st.number_input("Total Freight (BRL)", 0.0, 5000.0, 10.0)

        col4, col5 = st.columns(2)
        with col4:
            payment_total = st.number_input("Payment Total (BRL)", 0.0, 50000.0, 100.0)

        with col5:
            seller_order_count = st.number_input("Seller Order Count", 0, 500000, 10)

        submitted = st.form_submit_button("Predict Delivery Time")

    if submitted:
        try:
            input_data = {
                "purchase_hour": purchase_hour,
                "purchase_dow": purchase_dow,
                "purchase_month": purchase_month,
                "num_items": num_items,
                "total_price": total_price,
                "total_freight": total_freight,
                "payment_total": payment_total,
                "seller_order_count": seller_order_count
            }

            X = pd.DataFrame([np.zeros(len(model_cols))], columns=model_cols)

            for k, v in input_data.items():
                if k in X.columns:
                    X.loc[0, k] = v

            prediction = model.predict(X)[0]

            st.success(f" Predicted Delivery Time: **{prediction:.2f} days**")

        except Exception as e:
            st.error("Prediction failed.")
            st.text(traceback.format_exc())

# TAB 2 — FINDINGS & ANALYSIS
with tab2:

    st.header(" KPI Summary")

    # KPI Summary Cards
    k1, k2, k3 = st.columns(3)

    k1.metric(" Avg Delivery Time", f"{avg_delivery:.2f} days")
    k2.metric(" Fastest Delivery Month", fastest_month_name)
    k3.metric(" Slowest Delivery Month", slowest_month_name)

    k4, k5 = st.columns(2)
    k4.metric(" Avg Freight Value", f"R$ {avg_freight:.2f}")
    k5.metric(" Month With Most Orders", most_orders_month_name)

    st.markdown("---")

    # Feature importance PNG
    st.subheader("Feature Importance")
    try:
        st.image("Feature_imp_olist.png")
    except:
        st.warning("Feature_imp_olist.png not found.")

    st.markdown("---")

    st.subheader(" SHAP Summary Plot")
    try:
        st.image("shap_olist.png")
    except:
        st.warning("shap_olist.png not found.")

    st.markdown("---")

    st.header(" Interpretation Summary")
    st.write("""
### Key Insights:
- **Freight cost** is the strongest indicator of delivery delay  
- **Seller experience** significantly reduces delivery time  
- **Payment value and order price** affect priority  
- **Seasonality:** Certain months consistently show slower delivery performance  
""")

# TAB 3 — OPTIMIZATION STRATEGIES
with tab3:
    st.header(" Recommended Delivery Optimization Strategies")

    st.write("""
These recommendations are based on how the model behaves and what impacts delivery time the most.
""")

    optimizations = pd.DataFrame([
        ["Choose sellers with high order counts",
         "Experienced sellers deliver faster with fewer delays",
         "Low seller rating → higher delay risk"],

        ["Reduce freight distance",
         "Freight cost is the strongest predictor of long delivery time",
         "High freight values → far-away sellers"],

        ["Order earlier in the day",
         "Earlier purchases get processed on the same day",
         "Evening orders usually wait until next morning"],

        ["Avoid peak-season months",
         "Holidays and events cause predictable delays",
         "Black Friday, Christmas, Carnival"],

        ["Smaller item batches",
         "Few items → faster packaging and processing",
         "Large multi-item orders slow down fulfillment"],

        ["Prioritize higher-value orders",
         "High-value orders receive fulfillment priority",
         "Low-value orders may be queued longer"],

        ["Check seller location",
         "Nearby sellers reduce freight and delivery time",
         "Remote sellers → slower delivery"]
    ], columns=["Optimization Strategy", "Why It Helps", "When It Matters"])

    st.table(optimizations)
    
    st.markdown("---")

# TAB 4 — Q&A / MODEL INTERPRETATION
with tab4:

    st.header(" Model Interpretation & Key Findings")
    st.markdown("---")

    st.subheader("1. What the Model Learns")
    st.write("""
The model predicts **delivery time (in days)** using order attributes such as
freight, price, number of items, payment amount, purchase month, and seller experience.

It is a **regression model** that outputs a continuous value, optimized using MAE and RMSE.
""")

    st.subheader(" 2. Most Influential Features (What Matters Most)")
    st.write("""
Based on **Feature Importance** and **SHAP analysis**, the most important predictors are:

###  **1. Total Freight**
- Strongest predictor  
- Higher freight almost always means longer delivery time  
- Freight represents **distance + logistics complexity**

###  **2. Seller Order Count**
- Experienced sellers deliver faster  
- Inexperienced sellers → high variability in delivery time

###  **3. Payment Total & Total Price**
- High-value orders are typically prioritized  
- Low-value orders show more inconsistent delivery times

###  **4. Purchase Month (Seasonality)**
- Some months introduce predictable delays (holidays, sales events)

###  **5. Number of Items**
- Larger orders slightly increase delivery time due to packaging effort  
""")

    st.subheader(" 3. SHAP Interpretation (Why a Prediction Happens)")
    st.write("""
SHAP explains **how each feature moves the prediction up or down**.

###  Red points (high value)
Push the prediction **upwards** → longer delivery  
Example: high freight → strong delay influence  

###  Blue points (low value)
Push the prediction **downwards** → faster delivery  
Example: low freight → quicker delivery  

SHAP confirms:
- Freight is the dominant driver  
- Seller experience contributes to speed  
- Certain categories tend to have slower delivery patterns
""")

    st.subheader(" 4. Behavioral & Seasonal Insights")
    st.write("""
###  Purchase Month
- Delivery delays cluster around specific high-volume months  

###  Purchase Hour
- Evening purchases slightly delay fulfillment  
- Because processing generally starts the next morning  

###  Number of Items
- Slight contributor, but not as impactful as freight or seller performance  
""")

    st.subheader(" 5. Practical Conclusions")
    st.write("""
###  How to Reduce Delivery Time (Consumer)
- Choose sellers with high order counts  
- Avoid far-away sellers (high freight)  
- Place orders earlier in the day  
- Avoid peak-season months if urgent delivery is needed  

###  Business Recommendations
- Freight/logistics optimization yields biggest improvements  
- Improve seller scoring/monitoring for reliability  
- Plan inventory & staffing for seasonal peaks  
""")

    st.markdown("---")
    st.success("This section summarizes how the model works and explains delivery delays using data-driven insights.")


