
# 📊 Marketing Performance Optimization & Forecasting           

![Facebook vs AdWords](https://www.clematistech.com/wp-content/uploads/2021/08/Google-Ads-vs-FB-Ads-1.png)

# Problem Statement
A Marketing Agency wants to maximise the ROI(return on investment) for their client's advertising campaign.They conducted two Campaigns: one on Facebook and another on Adwords.We need to determine which Platform is better in terms of clicks, Conversions and overall cost effectiveness. By identifying the most effective platform, the Agency can allocate the resources more effectively and optimise the advertising strategies to deliver better outcomes for the clients

# 🚀 Project Overview

In the highly competitive landscape of digital marketing, businesses constantly strive to allocate their ad budgets more intelligently across platforms. This project presents a comprehensive data-driven evaluation of two major advertising giants — Facebook Ads and Google AdWords — to determine which platform delivers superior performance in terms of conversions, efficiency, and return on investment (ROI).

The core objective is to help marketers and analysts make informed, evidence-based decisions about where to invest advertising resources for maximum impact. By analyzing over 1,000 ad campaigns, this study delves into key performance metrics such as:

📈 Conversion Efficiency: How effectively do clicks and views turn into tangible actions like purchases, signups, or leads?

💵 Return on Investment (ROI): Which platform offers more value per advertising dollar spent?

🎯 Click-Through & Conversion Rates: What does user engagement look like post-ad exposure?

🔮 Forecasted Campaign Performance: What are the long-term trends in conversion rates and ad behavior?


# 🧠 Objectives

- 📈 Identify trends and patterns in ad performance
- 🔍 Investigate relationships between views, clicks, and conversions
- 🧪 Validate performance differences using statistical tests
- 🔮 Forecast future ad performance using Prophet
- 🤖 Predict conversions using regression models





# 📁 Dataset

- **Source**: [`Marketing-Campaign.csv`](./data/Marketing-Campaign.csv)
- **Records**: 1,000 ad campaigns
- **Description**: This dataset contains real-world-like performance data for Facebook and Google Ad campaigns. It is available in this GitHub repository under the `/data` folder.
  
You will find the full dataset [here on GitHub](https://github.com/Prabh10p/Ad-Campaign-Analytics-A-B-Testing-Forecasting-Conversion-Prediction).

  **Key Columns**:
  - `facebook_ad_views`, `facebook_ad_clicks`, `facebook_ad_conversions`
  - `adword_ad_views`, `adword_ad_clicks`, `adword_ad_conversions`
  - Additional metrics: `CTR`, `CPC`, `Conversion Rate`, and `Cost per Conversion`

# 🧠 Methodology
To extract deep, actionable insights, we implement a full-stack analytical pipeline combining modern data science techniques and marketing intelligence:

## 🔍 Exploratory Data Analysis (EDA)

  Distribution of conversions across platforms

  Relationship between views, clicks, and conversions
  
  ROI distribution and cost-effectiveness metrics
## 📊 Statistical Analysis

Calculation of means, medians, correlation coefficients, and standard deviation
Identification of patterns and anomalies in campaign performance

## 🧪 A/B Hypothesis Testing

Conducted independent two-sample t-tests to determine if Facebook Ads significantly outperform AdWords in conversion metrics
Verified statistical significance of observed differences

## 🔮 Time Series Forecasting

Utilized Facebook Prophet to forecast future conversion trends based on historical campaign performance
Integrated seasonalities (weekly, monthly, yearly) and campaign-specific holidays to enhance accuracy

## 🤖 Machine Learning Modeling

Applied Linear Regression models to predict expected conversions based on ad views and clicks
Evaluated performance using metrics like R², MAE, and MSE
# 📈 Results Summary

Following an in-depth analysis of 1,000 advertising campaigns across both Facebook and Google AdWords, the key findings are outlined below:

---

### 🔹 Conversion Performance

| Metric                   | Facebook Ads     | Google AdWords   |
|--------------------------|------------------|------------------|
| Average Conversions      | **62.34**         | 31.67            |
| Conversion Rate (%)      | **2× higher**     | Lower            |
| Conversion Consistency   | ✅ Stable          | ❌ Highly Variable |

- **Facebook outperformed AdWords by nearly double the average conversions per campaign.**
- **AdWords campaigns** showed inconsistent conversion patterns, with high click volume but lower post-click conversion rates.

---

### 🔹 Click Behavior & Engagement

| Metric                   | Facebook Ads     | Google AdWords   |
|--------------------------|------------------|------------------|
| Clicks per Campaign      | Lower             | **Higher**        |
| CTR (Click Through Rate) | Moderate          | **Higher**        |
| Engagement Quality       | ✅ High (intent-driven) | ❌ Low (generic reach) |

- Although AdWords generated more clicks and impressions, **Facebook’s clicks led to significantly more conversions**, indicating **higher user intent and better audience targeting**.

---

### 🔹 Return on Investment (ROI)

| Metric                      | Facebook Ads       | Google AdWords     |
|-----------------------------|--------------------|--------------------|
| CPC (Cost per Click)        | Higher              | Lower              |
| Cost per Conversion (CPCv)  | **Lower overall**   | Higher              |
| ROI                         | ✅ More efficient    | ❌ Less efficient   |

- Facebook achieved **greater ROI** despite slightly higher CPC, due to its stronger conversion performance.
- AdWords campaigns often consumed budget without corresponding conversion gains.

---

### 🔹 Forecasting & Predictive Modeling

- 📈 **Linear Regression** model (for Facebook):
  - **R² Score**: 0.85
  - **MAE**: 4.90
  - **MSE**: 3.86
  - ✅ Indicates a strong linear relationship between clicks/views and conversions

- 🔮 **Prophet Time Series Forecasting** (for Facebook):
  - Forecasted conversions remained **stable and seasonally cyclical**
  - Spikes aligned with holidays and weekends
  - 🧠 Suggests **timing plays a critical role in campaign success**

---

### 📊 Statistical Significance

- **A/B Hypothesis Testing (t-test)**:
  - **t-statistic** ≈ 66  
  - **p-value** ≪ 0.05

✅ This confirms that the difference in conversion performance between Facebook and AdWords is **statistically significant**, and **not due to random variation**.

---

> 🔍 In summary: While AdWords excels in reach and visibility, Facebook delivers **higher conversion quality**, **better ROI**, and **predictable performance** — making it the more effective platform for campaigns focused on measurable outcomes.

# 📌 Key Insights

- **Facebook Ads** outperform AdWords in conversion quality
- **AdWords** have greater reach but lower conversion efficiency
- **Clicks and views** are strong predictors for conversions on Facebook
- **Seasonality, content quality, and timing** all affect outcomes

---

# 📣  Final Recommendations

Based on the comprehensive analysis of Facebook and AdWords campaigns, the following strategic recommendations are provided to the marketing agency to help maximize ROI and optimize advertising outcomes for their clients:

🟢 **Prioritize Facebook Ads for Higher Conversion Efficiency**  
   Facebook consistently demonstrated a stronger conversion rate, meaning a higher percentage of users who click actually take action (purchase, sign-up, etc.). This indicates a more effective funnel and suggests that **Facebook should receive a greater share of the ad budget**, especially for performance-driven campaigns.

💰 **Invest Where ROI is Strongest — Facebook for Cost-Effectiveness**  
   Despite sometimes having a higher cost per click, Facebook achieved better ROI by delivering more conversions per dollar spent. This highlights that **the platform’s targeting and engagement are more aligned with user intent**, making it a smarter long-term investment.

🔁 **Use Google AdWords Strategically for Reach-Based Campaigns**  
   AdWords showed higher visibility and click volume but lower conversion follow-through. It is better suited for **brand awareness campaigns or top-of-funnel goals** rather than immediate conversion. Limit AdWords budget to campaigns where **reach or exposure is more important than direct conversions**.

📊 **Implement Data-Driven Budget Allocation**  
   Shift toward a **performance-based budgeting model**. Use conversion metrics and ROI data to guide platform spend, rather than allocating budgets evenly. Campaign-level optimization should be an ongoing practice rather than a one-time decision.

 🔄 **Optimize Underperforming Campaigns — Especially AdWords**  
   Before scaling AdWords spend, audit and refine key components:
   - Improve audience targeting parameters
   - Adjust keyword strategy
   - Test more compelling ad copy and creative
   - Align landing pages with user intent

📅 **Launch Campaigns Around Peak Conversion Periods**  
   Time series forecasting revealed that certain time windows (e.g., weekends, holidays) produce more conversions. Align future Facebook campaigns with these high-performing intervals to **capitalize on user intent spikes**.

🤖 **Use Predictive Modeling to Estimate Returns Before Launch**  
   Leverage regression models and Prophet forecasting to **simulate campaign performance before committing spend**. Predictive insights can significantly reduce risk and increase strategic clarity.

🧪 **Maintain Ongoing A/B Testing to Validate Strategies**  
   Continue running A/B tests across both platforms to evaluate new creatives, audience segments, and budget adjustments. This ensures continuous learning and adaptation to shifting user behaviors and platform algorithm changes.

---

📌 By reallocating budget based on proven conversion efficiency, optimizing underperforming campaigns, and forecasting intelligently, the marketing agency can **maximize client ROI, reduce wasted spend, and significantly improve advertising outcomes.**


# 👨‍💻 Author

**Prabhjot Singh**  
🎓 B.S. in Information Technology, Marymount University  
🔗 [LinkedIn](https://www.linkedin.com/in/prabhjot-singh-10a88b315/)  
🧠 Passionate about data-driven decision making, analytics, and automation

---

## ⭐ Support

If you found this project useful, feel free to ⭐ it and share it with others!
