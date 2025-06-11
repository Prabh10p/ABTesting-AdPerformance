import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# Set default page
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Home"

# Page config
st.set_page_config(
    page_title="Ad Campaign Marketing Analytics and Forecasting",
    page_icon="images.png",
    layout="wide",
    initial_sidebar_state="auto"
)

# Sidebar navigation
with st.sidebar:
    st.title("Ad Campaign Marketing Analytics and Forecasting")
    st.image("d5f69539-87e7-4356-97b0-de548cd7b84a.png")
    st.markdown("Built by **Prabhjot Singh**")
    st.caption("A dashboard comparing AdWords and Facebook Ads with forecasting and regression analysis")
    st.markdown("### Navigation")
    if st.button("Home"):
        st.session_state.active_page = "Home"
    if st.button("Analytics Dashboard"):
        st.session_state.active_page = "Analytics Dashboard"
    if st.button("Forecasting Dashboard"):
        st.session_state.active_page = "Forecasting Dashboard"
    if st.button("Regression Analysis"):
        st.session_state.active_page = "Regression Analysis"

# Load dataset
url = "Marketing-Campaign.csv"
df = pd.read_csv(url)
df["date_of_campaign"] = pd.to_datetime(df["date_of_campaign"])
df.rename(columns={"date_of_campaign": "date"}, inplace=True)
future_clicks = df.head(1365)
df = df.head(1000)
df_forecast = df.copy()

# Page content rendering
if st.session_state.active_page == "Home":
    st.title("📊 Marketing Data Overview")
    st.markdown("A visual and statistical overview of your marketing campaign data.")
    img1 = Image.open("0304_Best_Data_Lina_Newsletter___blog-2be50c35d82d6815cfbb3b8b5d38f6d3__1_.png")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(img1, use_container_width=True)

    st.markdown("### 📋 Data Exploration")

    # Cleaner UI with expanders
    with st.expander("🔍 Show Raw Data Sample"):
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📑 Show All Column Names"):
        st.dataframe(pd.DataFrame(df.columns, columns=["Columns"]), use_container_width=True)

    with st.expander("🧠 Show Data Types"):
        dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
        st.dataframe(dtype_df, use_container_width=True)

    with st.expander("📈 Show Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    st.success("Use the sidebar to switch between analysis views.")

elif st.session_state.active_page == "Analytics Dashboard":
    st.subheader("📊 Facebook vs AdWords Mean Ad Performance")

    metrics = {
        "Views": ("facebook_ad_views", "adword_ad_views"),
        "Clicks": ("facebook_ad_clicks", "adword_ad_clicks"),
        "Conversions": ("facebook_ad_conversions", "adword_ad_conversions"),
        "Cost per Ad": ("facebook_cost_per_ad", "adword_cost_per_ad"),
        "CTR": ("facebook_ctr", "adword_ctr"),
        "Conversion Rate": ("facebook_conversion_rate", "adword_conversion_rate")
    }

    cols = st.columns(3)
    for idx, (label, (fb_col, ad_col)) in enumerate(metrics.items()):
        fig = px.bar(
            x=["Facebook", "AdWords"],
            y=[df[fb_col].mean(), df[ad_col].mean()],
            title=f"Average {label}",
            text_auto=True,
            labels={"x": "Platform", "y": label},
            color=["Facebook", "AdWords"],
            color_discrete_sequence=["#1877F2", "#FF9900"]  # Blue for Facebook, Orange for AdWords
        )
        cols[idx % 3].plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Facebook vs AdWords Total Ad Performance")
    cols1 = st.columns(3)
    for idx1, (label1, (fb_col1, ad_col1)) in enumerate(metrics.items()):
        fig1 = px.bar(
            x=["Facebook", "AdWords"],
            y=[df[fb_col1].sum(), df[ad_col1].sum()],
            title=f"Total {label1}",
            text_auto=True,
            labels={"x": "Platform", "y": label1},
            color=["Facebook", "AdWords"],
            color_discrete_sequence=["#1877F2", "#FF9900"]  # Blue for Facebook, Orange for AdWords
        )
        cols1[idx1 % 3].plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Facebook vs AdWords Cost Per Conversion")
    ax = px.bar(
        x=["Facebook(CPC)", "Adwords(CPC)"],
        y=[df["facebook_cost_per_ad"].sum() / df["facebook_ad_conversions"].sum(),
           df["adword_cost_per_ad"].sum() / df["adword_ad_conversions"].sum()],
        text_auto=True,
        color=["#1877F2", "#FF9900"])
    st.plotly_chart(ax, use_container_width=True)

    st.subheader("📊 frequency of High conversion Day as compare to Low conversion Days")


    def conversion(value):
        values = []
        for i in value:
            if i < 20:
                values.append("Low Conversion")
            elif 20 <= i <= 40:
                values.append("Medium Conversion")
            else:
                values.append("High Conversion")
        return values


    # Assume 'conversion' is a function that categorizes conversion counts into labels
    df["f_conversion"] = conversion(df["facebook_ad_conversions"])
    df["ad_conversion"] = conversion(df["adword_ad_conversions"])

    # Frequency counts
    f_freq = df["f_conversion"].value_counts().reset_index()
    f_freq.columns = ["Category", "Facebook"]

    a_freq = df["ad_conversion"].value_counts().reset_index()
    a_freq.columns = ["Category", "AdWords"]

    # Merge both on Category
    merged_freq = pd.merge(f_freq, a_freq, on="Category", how="outer").fillna(0)

    # Melt to long-form for Plotly
    df_melted = pd.melt(merged_freq, id_vars="Category", value_vars=["Facebook", "AdWords"],
                        var_name="Platform", value_name="Count")

    # Plot
    fig = px.bar(
        df_melted,
        x="Category",
        y="Count",
        color="Platform",
        barmode="group",
        text_auto=True,
        color_discrete_map={"Facebook": "#1877F2", "AdWords": "#FF9900"},
        title="Conversion Category Frequency by Platform"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Facebook vs AdWords: Clicks vs. Conversions")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Increased height for better spacing

    # Facebook Scatterplot
    sns.scatterplot(x=df["facebook_ad_clicks"], y=df["facebook_ad_conversions"], ax=ax1, color="#1877F2")
    ax1.set_title("Facebook Click and Conversions")
    ax1.set_xlabel("Facebook Clicks")
    ax1.set_ylabel("Facebook Conversions")

    # AdWords Scatterplot
    sns.scatterplot(x=df["adword_ad_clicks"], y=df["adword_ad_conversions"], ax=ax2, color="#FF9900")
    ax2.set_title("AdWords Click and Conversions")
    ax2.set_xlabel("AdWords Clicks")
    ax2.set_ylabel("AdWords Conversions")
    st.pyplot(fig)

    st.subheader("Facebook and Adword Correelation")
    fig1, ax = plt.subplots(1, figsize=(10, 6))
    df_new = df.copy()
    df_new.drop(["date", "f_conversion", "ad_conversion"], axis=1, inplace=True)
    sns.heatmap(df_new.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig1)


elif st.session_state.active_page == "Forecasting Dashboard":

    from prophet import Prophet

    df_forecast = df[["date", "facebook_ad_conversions", "facebook_ad_clicks"]]
    df_forecast["date"] = pd.to_datetime(df_forecast["date"])
    df_forecast.rename({"date": "ds", "facebook_ad_conversions": "y"}, axis=1, inplace=True)

    split_window = int(len(df_forecast) * 0.80)
    df_train = df_forecast[:split_window]
    df_test = df_forecast[split_window:]

    # Train model
    model2 = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model2.add_regressor('facebook_ad_clicks')

    model2.fit(df_train)

    future = df_test[['ds', 'facebook_ad_clicks']].copy()
    forecast = model2.predict(future)

    import plotly.graph_objects as go

    st.subheader('Actual vs Predicted Facebook Conversions Over Time')
    # --- Plot Actual vs Predicted ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test['ds'], y=df_test['y'],
        mode='lines+markers', name='Actual',
        line=dict(color='black', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_test['ds'], y=forecast['yhat'],
        mode='lines+markers', name='Predicted',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Faecebook Conversions',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecasted Facebook Conversion For Next Year")
    # --- Forecasting for Next 365 Days ---
    future_dates = pd.date_range(start=df_test["ds"].iloc[-1] + pd.Timedelta(days=1), periods=365, freq="D")

    # Ensure future_clicks is 1D
    future = future_clicks["facebook_ad_clicks"].values[-365:]

    df_new = pd.DataFrame({
        "ds": future_dates,
        "facebook_ad_clicks": future
    })

    forecast_365 = model2.predict(df_new)

    # --- Forecast Plot ---
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_test['ds'], df_test['y'], label='Actual', color='black')
    ax.plot(forecast_365['ds'], forecast_365['yhat'], label='Forecast', color='green')
    ax.legend()
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    st.pyplot(fig2)

elif st.session_state.active_page == "Regression Analysis":
    st.subheader("🔮 Predict Facebook Conversions")

    clicks = st.slider("Number of Clicks", 0, 100000, 300)

    scaler = pickle.load(open("scalerfinal.pkl", "rb"))
    model_final = pickle.load(open("modelfinal.pkl", "rb"))

    if st.button("🎯 Predict Now"):
        try:
            input_vals = np.array([[clicks]])
            scaled = scaler.transform(input_vals)
            result = model_final.predict(scaled)
            st.success(f"✅ Predicted Facebook Conversions: `{int(result[0])}`")
        except Exception as e:
            st.error(f" Prediction error: {e}")

    # ----------------- Model Benchmark -----------------
    st.subheader("📉 Model Benchmarking")

    X = df[["facebook_ad_clicks"]]
    y = df["facebook_ad_conversions"]

    scaler_bench = StandardScaler()
    X_scaled = scaler_bench.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
    }

    for name, model_obj in models.items():
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.markdown(f"""
        #### 🔍 {name}
        - RMSE: `{rmse:.2f}`
        - MAE: `{mae:.2f}`
        - R²: `{r2:.2f}`
        """)

        if name == "Linear Regression":
            fig = px.scatter(title="Linear Regression: Actual vs Predicted")
            fig.add_scatter(x=list(range(len(y_test))), y=y_test.values, mode='markers', name='Actual')
            fig.add_scatter(x=list(range(len(y_test))), y=y_pred, mode='markers', name='Predicted')
            st.plotly_chart(fig, use_container_width=True)




