import streamlit as st
import pickle
import numpy as np
scaler = pickle.load(open('scaler4.pkl', 'rb'))
model = pickle.load(open("model1.pkl", 'rb'))

st.title("📈 Facebook Ads Conversion Predictor")

sms = st.text_input("Enter number of clicks and views (comma-separated)")

if st.button("Predict"):
    try:
        # Expect input like: "300,1200"
        input_vals = np.array([float(i) for i in sms.split(",")]).reshape(1, -1)
        scaled = scaler.transform(input_vals)
        result = model.predict(scaled)
        st.success(f"Predicted Conversions: {int(result[0])}")
    except Exception as e:
        st.error("❌ Invalid input. Please enter two numbers separated by a comma.")



# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("📊 A/B Testing: Facebook vs AdWords Ad Campaign Analytics")

# Upload CSV
url = "https://raw.githubusercontent.com/Prabh10p/ABTesting-AdPerformance/main/Marketing-Campaign.csv"



if url:
    df = pd.read_csv(url)
    df = df.head(1000)
    df["date_of_campaign"] = pd.to_datetime(df["date_of_campaign"])
    df.rename(columns={"date_of_campaign": 'date'}, inplace=True)
    st.success("Data loaded successfully!")

    if st.checkbox("Show Data Sample"):
        st.dataframe(df.head())

    # ------------------------ 📈 Plot Mean Comparisons ------------------------
    st.subheader("📊 Facebook vs AdWords Performance Comparison")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    sns.barplot(x=["Facebook", "Adword"],
                y=[df["facebook_ad_views"].mean(), df["adword_ad_views"].mean()],
                ax=axs[0, 0]).bar_label(axs[0, 0].containers[0])
    axs[0, 0].set_title("Mean Views")

    sns.barplot(x=["Facebook", "Adword"],
                y=[df["facebook_ad_clicks"].mean(), df["adword_ad_clicks"].mean()],
                ax=axs[0, 1]).bar_label(axs[0, 1].containers[0])
    axs[0, 1].set_title("Mean Clicks")

    sns.barplot(x=["Facebook", "Adword"],
                y=[df["facebook_ad_conversions"].mean(), df["adword_ad_conversions"].mean()],
                ax=axs[0, 2]).bar_label(axs[0, 2].containers[0])
    axs[0, 2].set_title("Mean Conversions")

    sns.barplot(x=["Facebook", "Adword"],
                y=[df["facebook_cost_per_ad"].mean(), df["adword_cost_per_ad"].mean()],
                ax=axs[1, 0]).bar_label(axs[1, 0].containers[0])
    axs[1, 0].set_title("Cost per Ad")

    sns.barplot(x=["Facebook", "Adword"],
                y=[df["facebook_ctr"].mean(), df["adword_ctr"].mean()],
                ax=axs[1, 1]).bar_label(axs[1, 1].containers[0])
    axs[1, 1].set_title("CTR")

    sns.barplot(x=["Facebook", "Adword"],
                y=[df["facebook_conversion_rate"].mean(), df["adword_conversion_rate"].mean()],
                ax=axs[1, 2]).bar_label(axs[1, 2].containers[0])
    axs[1, 2].set_title("Conversion Rate")

    plt.tight_layout()
    st.pyplot(fig)

    # ------------------------ 🤖 Model Evaluation ------------------------
    st.subheader("🤖 Model Performance: Predicting Facebook Conversions")

    X = df[["facebook_ad_clicks", "facebook_ad_views"]]
    y = df["facebook_ad_conversions"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"📌 **{name}**:  \n RMSE: `{rmse:.2f}`, MAE: `{mae:.2f}`, R²: `{r2:.2f}`")

        # Plot only for Linear Regression
        if name == "LinearRegression":
            plt.figure(figsize=(8, 4))
            plt.scatter(range(len(y_test)), y_test, alpha=0.5, color="green", label="Actual")
            plt.scatter(range(len(y_test)), y_pred, color="red", label="Predicted")
            plt.title("Linear Regression: Predicted vs Actual")
            plt.xlabel("Sample Index")
            plt.ylabel("Conversions")
            plt.legend()
            plt.grid(True)
            st.pyplot()

