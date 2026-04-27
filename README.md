# 📊 Data Cleaning & Visualization Project

## 📌 Objective

The goal of this project is to clean, process, and analyze an e-commerce dataset to extract meaningful insights using Python.

---

## 🛠 Tools & Technologies

* Python
* Pandas
* Matplotlib
* Seaborn

---

## 🧹 Data Cleaning Steps

* Removed duplicate records (3 rows)
* Handled missing values (7 entries filled)
* Converted data types (date formatting)
* Removed outliers using IQR method

---

## ⚙️ Feature Engineering

* Created `total_sales` column
* Extracted `month` and `day_of_week`

---

## 📊 Visualizations

The project includes 8 visualizations:

* Sales by category (bar chart)
* Sales trend over time (line chart)
* Category distribution (donut chart)
* Top-performing products (horizontal bar)
* Sales by city (bar chart)
* Correlation heatmap
* Sales distribution (box plot)
* Combined dashboard (2×2 layout)

---

## 🔍 Key Insights

* **Clothing is the highest revenue-generating category**, indicating strong customer demand
* **Mumbai contributes the most to total sales**, making it the top-performing city
* **UPI is the most frequently used payment method**, showing a preference for digital payments
* **Sales peak in March**, suggesting possible seasonal trends or promotional impact
* **Furniture has the highest average order value**, meaning fewer but higher-value purchases

---

## ✅ Conclusion

The dataset was successfully cleaned and analyzed. Visualizations helped uncover patterns and trends that can support business decision-making.

---

## 📂 Project Structure

```
data-analysis-project/
│
├── data/
│   ├── raw_data.csv
│   └── cleaned_data.csv
│
├── notebook/
│   └── analysis.ipynb
│
├── scripts/
│   └── data_analysis.py
│
├── images/
│   ├── chart1_sales_by_category.png
│   ├── chart2_monthly_revenue.png
│   ├── chart3_payment_distribution.png
│   ├── chart4_top_products.png
│   ├── chart5_sales_by_city.png
│   ├── chart6_correlation_heatmap.png
│   ├── chart7_sales_distribution.png
│   └── chart8_dashboard.png
│
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the script:
   ```
   python scripts/data_analysis.py
   ```

---

## 📌 Author

Harini S
