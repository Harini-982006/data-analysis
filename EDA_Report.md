# 📊 Exploratory Data Analysis (EDA) Report: E-Commerce Sales Performance

**Author:** Harini S  
**Date:** May 2026  
**Dataset:** E-Commerce Sales Data (Indian Market)  
**Tools:** Python (Pandas, Matplotlib, Seaborn), HTML/JS (Chart.js, PapaParse)

---

## 📋 1. Executive Summary

This report presents a comprehensive Exploratory Data Analysis (EDA) of the e-commerce sales dataset. The primary objective is to analyze transaction patterns, identify key factors driving revenue, and detect trends across different categories, cities, months, and payment methods. 

### Key Findings:
1. **Core Revenue Drivers**: **Clothing** is the highest revenue-generating category (₹48,566), accounting for 34.8% of the total revenue, while **Furniture** exhibits the highest Average Order Value (AOV) of **₹3,827.27** due to higher unit prices.
2. **Geographical Hub**: **Mumbai** is the dominant city in terms of total sales revenue, contributing **₹39,186** (28.0% of total sales).
3. **Payment Preferences**: **UPI** is the most preferred payment method, used in **23 orders** (45.1% of transactions), indicating strong adoption of mobile-first digital payment systems.
4. **Seasonal Peak**: Sales peaked significantly in **March 2024 (₹53,180)**, suggesting seasonal buying behavior or the impact of promotional campaigns.
5. **Data Quality**: The raw dataset had several anomalies—including duplicates, missing values, and extreme outliers (such as a power strip priced at ₹85,000)—which were successfully resolved during preprocessing to ensure reliable analysis.

---

## 🧹 2. Data Cleaning & Preprocessing

The raw dataset contained **63 rows** and **9 columns**. To perform a reliable analysis, a rigorous data cleaning pipeline was executed:

* **Duplicate Removal**: Identified and removed **3 duplicate records** (Orders 1005, 1012, and 1033), which were identical repeats.
* **Missing Value Imputation**:
  * **Customer Name**: Filled 2 missing values with *"Unknown Customer"* (Orders 1008 and 1038).
  * **Category**: Filled 2 missing values with *"Unknown"* (Orders 1022 and 1057).
  * **Quantity**: Filled 3 missing values with the **median quantity (1)** (Orders 1003, 1011, and 1035).
* **Outlier Treatment (IQR Method)**:
  * Calculated the Interquartile Range (IQR) for prices. The normal price range was determined to be **[₹-2,402, ₹6,601]**.
  * **Anomalous Prices**: Removed Order 1051 (Power Strip listed at **₹85,000**), which was an obvious data entry error.
  * **High-Value Outliers**: Removed 6 high-value transactions with prices above the upper limit (₹6,601), including Office Chair (₹8,999), TV Unit (₹7,500), Dining Table (₹12,000), Wardrobe (₹15,000), Dressing Table (₹6,800), and Recliner (₹22,000) to examine normal sales distributions.
  * **Negative Prices**: Removed Order 1052 (Graphic Tee listed at **-₹500**).
  * **Zero Quantity**: Removed Order 1054 (Webcam listed with **0 quantity**).
* **Feature Engineering**:
  * Created a `total_sales` column calculated as `price * quantity`.
  * Extracted temporal features: `month`, `month_name`, and `day_of_week` from the transaction date.

**Final Cleaned Dataset**: **51 rows** and **13 columns**.

---

## 📊 3. Descriptive Statistics

A statistical summary of the numerical attributes in the cleaned dataset reveals the following:

| Metric | Unit Price (₹) | Quantity | Total Sales (₹) |
| :--- | :---: | :---: | :---: |
| **Total Sum** | - | 97 | **₹1,39,719** |
| **Mean** | ₹1,959.04 | 1.90 | ₹2,739.59 |
| **Median** | ₹1,800.00 | 1.00 | ₹2,499.00 |
| **Standard Deviation** | ₹1,298.80 | 1.54 | ₹1,264.45 |
| **Minimum** | ₹299.00 | 1.00 | ₹399.00 |
| **Maximum** | ₹5,600.00 | 10.00 | ₹5,600.00 |

### Key Observations:
* The **Average Order Value (AOV)** is **₹2,739.59**.
* The **Median Order Value** is **₹2,499.00**, indicating that half of the transactions are under ₹2,500.
* A high standard deviation in prices (₹1,298.80) and sales (₹1,264.45) reflects a diverse product catalog ranging from inexpensive accessories (₹299) to mid-range furniture items (₹5,600).

---

## 🔗 4. Correlation & Influencing Factors

To understand the relationships between numerical variables, a Pearson correlation matrix was computed:

| Variable | Price | Quantity | Month | Total Sales |
| :--- | :---: | :---: | :---: | :---: |
| **Price** | 1.00 | -0.50 | 0.06 | **0.71** |
| **Quantity** | -0.50 | 1.00 | 0.00 | **0.04** |
| **Month** | 0.06 | 0.00 | 1.00 | 0.07 |
| **Total Sales** | **0.71** | **0.04** | 0.07 | 1.00 |

### Insights on Influencing Factors:
1. **Price is the Primary Driver of Revenue ($r = 0.71$)**:
   There is a strong, positive correlation between unit price and total sales value. Higher-ticket items contribute significantly to the total revenue, even when purchased in smaller quantities.
2. **Quantity and Price are Moderately Negatively Correlated ($r = -0.50$)**:
   As price increases, the quantity ordered tends to decrease. This aligns with standard consumer demand behavior: customers buy higher-priced items (like Coffee Tables or Blazers) in single units (typically 1 per order), whereas cheaper items (like Pen Drives or Mouse Pads) are often purchased in higher quantities (up to 10 per order).
3. **Quantity Has Minimal Correlation with Total Sales ($r = 0.04$)**:
   An increase in transaction volume (quantity) has almost no linear correlation with total sales value in this cleaned dataset. This is because high-volume orders are concentrated on lower-priced items (e.g., a Pen Drive order with quantity 10 totals ₹3,990, which is comparable to a single high-priced purchase).

---

## 📈 5. Detailed Segment Analysis

### A. Sales by Category
* **Clothing**: Generates **₹48,566** (34.8% of total revenue) across 18 transactions, making it the volume and revenue leader.
* **Electronics**: Generates **₹44,955** (32.2% of total revenue) across 20 transactions.
* **Furniture**: Generates **₹42,100** (30.1% of total revenue) across 11 transactions. It has the highest Average Order Value (**₹3,827.27**), showing high profitability per order.
* **Unknown**: Accounts for ₹4,098 (2.9% of total revenue).

### B. Geographical Distribution (Sales by City)
* **Mumbai**: Leads the market with **₹39,186** in revenue, establishing itself as the top-performing city.
* **Bangalore**: Follows with **₹26,482** in revenue.
* **Hyderabad**: Contributes **₹24,891**.
* **Delhi**: Contributes **₹18,987**.
* **Kolkata**: Contributes **₹17,979**.
* **Chennai**: Contributes **₹12,194**.

*Takeaway: Western and Southern cities (Mumbai, Bangalore, Hyderabad) are the strongest revenue hubs, contributing over 64% of total sales.*

### C. Monthly Performance Trends (Jan - Apr 2024)
* **January**: ₹38,679
* **February**: ₹33,771
* **March**: **₹53,180** *(Peak Month)*
* **April**: ₹14,089 *(Partial month or steep drop)*

*Takeaway: Sales show a clear seasonal peak in March, possibly due to tax year-end clearance, spring sales, or seasonal shifts. April shows a decline, which could represent incomplete reporting for that month.*

### D. Payment Method Adoption
* **UPI**: The most popular payment method with **23 orders** (45.1% share).
* **Credit Card**: Used in **14 orders** (27.5% share).
* **Debit Card**: Used in **11 orders** (21.6% share).
* **EMI**: Used in **3 orders** (5.9% share) in this cleaned dataset (since high-value furniture EMI orders were removed as outliers).

---

## 💡 6. Actionable Business Recommendations

Based on the findings, the following strategies are recommended:

1. **Leverage the High AOV of Furniture**:
   * Since Furniture has the highest average order value (₹3,827.27), bundle related items (e.g., Coffee Table with coasters or small cushions) to increase cart sizes.
   * Provide flexible payment terms and spotlight EMI options on the furniture category landing pages.
2. **Optimize Marketing Spend in Top Cities**:
   * Focus digital advertising budgets on **Mumbai, Bangalore, and Hyderabad**, which represent the core revenue hubs.
   * Tailor marketing campaigns to local regional trends (e.g., IT lifestyle tech accessories in Bangalore, fashion clothing in Mumbai).
3. **Promote UPI Payments**:
   * Encourage UPI usage through small incentives (e.g., 1-2% instant cashbacks or free shipping). This reduces credit card transaction fees for the business while aligning with the customers' primary payment method.
4. **Capitalize on March Seasonal Peaks**:
   * Investigate the exact causes of the March peak (promotional campaigns, financial year-end bonuses, etc.) and plan mini-campaigns in slower months (like February or April) to stabilize monthly revenue.
5. **Implement Data Validation Filters**:
   * Introduce automated validation on the e-commerce check-out/admin system to reject negative prices, zero quantities, or extreme values (e.g., ₹85,000 for a power strip) before they enter the database.
