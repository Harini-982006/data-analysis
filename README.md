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
│   ├── cleaned_data.csv
│   └── ml_results.csv
│
├── notebook/
│   ├── analysis.ipynb
│   └── ml_model.ipynb
│
├── scripts/
│   ├── data_analysis.py
│   └── ml_model.py
│
├── images/
│   ├── chart1_sales_by_category.png
│   ├── ...
│   ├── chart8_dashboard.png
│   ├── ml_chart1_accuracy_comparison.png
│   ├── ml_chart2_confusion_matrices.png
│   ├── ml_chart3_roc_curves.png
│   ├── ml_chart4_feature_importance.png
│   └── ml_chart5_metrics_comparison.png
│
├── README.md
├── requirements.txt
└── run_dashboard.py
```

---

## 🤖 Predictive Modeling (Machine Learning)

Applied supervised machine learning algorithms to classify orders into target categories based on pricing, quantity, and total sales.

* **Algorithms Evaluated**: Logistic Regression, Decision Tree, Random Forest
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrices, ROC Curves
* **Best Model**: **Random Forest** (~53.8% accuracy on 75/25 stratified split)
* **Results**: Exported to `data/ml_results.csv` and visual charts saved to `images/`

---

## ▶️ How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Analysis & Cleaning**:
   ```bash
   python scripts/data_analysis.py
   ```

3. **Run ML Predictive Modeling**:
   ```bash
   python scripts/ml_model.py
   ```

4. **Launch Interactive Dashboard**:
   ```bash
   python run_dashboard.py
   ```

---

## 📌 Author

Harini S
