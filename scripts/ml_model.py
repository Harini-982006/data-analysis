# ============================================================
# 🤖 PREDICTIVE MODELING USING MACHINE LEARNING
# ============================================================
# Author : Harini S
# Date   : May 2026
# Task   : Build a model to predict outcomes based on given data
# Tools  : Scikit-learn, Pandas, Matplotlib, Seaborn
# ============================================================

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────

import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ── Project Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

print("=" * 60)
print("🤖 PREDICTIVE MODELING USING MACHINE LEARNING")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 2: LOAD THE CLEANED DATASET
# ─────────────────────────────────────────────────────────────

df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_data.csv"))

print(f"\n✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 3: DEFINE THE PREDICTION TARGET
# ─────────────────────────────────────────────────────────────
# Goal: Predict the product CATEGORY based on features like
#       price, quantity, city, payment_method, month, day_of_week
# This is a CLASSIFICATION problem (supervised learning)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🎯 STEP 3: DEFINING PREDICTION TARGET")
print("=" * 60)

# Target variable
TARGET = 'category'

# Remove 'Unknown' category (too few samples for reliable ML)
df = df[df[TARGET] != 'Unknown'].reset_index(drop=True)
print(f"\n   Target Variable: '{TARGET}'")
print(f"   Task Type: Multi-class Classification")
print(f"   (Removed 'Unknown' category — insufficient samples for ML)")
print(f"\n   Class Distribution:")
class_dist = df[TARGET].value_counts()
for cat, count in class_dist.items():
    print(f"      {cat:15s} → {count} samples ({count/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 4: FEATURE ENGINEERING & ENCODING
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("⚙️ STEP 4: FEATURE ENGINEERING & ENCODING")
print("=" * 60)

# Select features for the model
feature_cols = ['price', 'quantity', 'city', 'payment_method', 'month', 'day_of_week']
print(f"\n   Selected Features: {feature_cols}")

# Encode categorical features
label_encoders = {}
df_ml = df.copy()

categorical_features = ['city', 'payment_method', 'day_of_week']
for col in categorical_features:
    le = LabelEncoder()
    df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le
    print(f"   → Encoded '{col}': {list(le.classes_)}")

# Encode target variable
le_target = LabelEncoder()
df_ml['target_encoded'] = le_target.fit_transform(df_ml[TARGET].astype(str))
print(f"\n   → Target classes: {list(le_target.classes_)}")

# Prepare feature matrix X and target vector y
X = df_ml[['price', 'quantity', 'month',
           'city_encoded', 'payment_method_encoded', 'day_of_week_encoded']]
y = df_ml['target_encoded']

print(f"\n   Feature Matrix (X): {X.shape}")
print(f"   Target Vector (y): {y.shape}")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 5: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("✂️ STEP 5: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n   Training Set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Testing Set : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   → Features scaled using StandardScaler ✅")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 6: TRAIN MACHINE LEARNING MODELS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🧠 STEP 6: TRAINING ML MODELS")
print("=" * 60)

# Dictionary to store models and results
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n── Training: {name} ──")

    # Use scaled features for Logistic Regression, raw for tree-based
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

    print(f"   Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1-Score  : {f1:.4f}")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 7: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("📊 STEP 7: MODEL COMPARISON")
print("=" * 60)

print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 67)
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1_score']:>10.4f}")

# Find best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_acc = results[best_model_name]['accuracy']
print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_acc*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 8: DETAILED CLASSIFICATION REPORT (Best Model)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"📋 STEP 8: CLASSIFICATION REPORT — {best_model_name}")
print("=" * 60)

best_y_pred = results[best_model_name]['y_pred']
# Use labels parameter to match only classes present in the data
all_labels = sorted(y_test.unique())
report = classification_report(y_test, best_y_pred,
                               labels=all_labels,
                               target_names=[le_target.classes_[i] for i in all_labels],
                               zero_division=0)
print(f"\n{report}")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 9: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("📊 STEP 9: GENERATING ML VISUALIZATIONS...")
print("=" * 60)

colors_ml = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a']

# ════════════════════════════════════════════════════════════
# FIGURE 1: MODEL ACCURACY COMPARISON (Bar Chart)
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
bar_colors = ['#3b82f6', '#10b981', '#f59e0b']

bars = ax.bar(model_names, accuracies, color=bar_colors,
              edgecolor='white', linewidth=2, width=0.5, zorder=3)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=13, color='#1e293b')

ax.set_title('🏆 Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(accuracies) + 15)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'ml_chart1_accuracy_comparison.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print("   ✅ ML Chart 1: Model Accuracy Comparison — saved!")

# ════════════════════════════════════════════════════════════
# FIGURE 2: CONFUSION MATRICES (All 3 Models)
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('🔍 Confusion Matrices — All Models', fontsize=16, fontweight='bold', y=1.02)

cmaps = ['Blues', 'Greens', 'Oranges']
# Get only the classes present in the test set
cm_labels = sorted(y_test.unique())
cm_class_names = [le_target.classes_[i] for i in cm_labels]

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'], labels=cm_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[idx],
                xticklabels=cm_class_names,
                yticklabels=cm_class_names,
                ax=axes[idx], cbar=False,
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                linewidths=1.5, linecolor='white')
    axes[idx].set_title(name, fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=10, fontweight='bold')
    axes[idx].tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'ml_chart2_confusion_matrices.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print("   ✅ ML Chart 2: Confusion Matrices — saved!")

# ════════════════════════════════════════════════════════════
# FIGURE 3: ROC CURVES (One-vs-Rest for each class)
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 8))
line_styles = ['-', '--', '-.']
model_colors = ['#3b82f6', '#10b981', '#f59e0b']

from sklearn.preprocessing import label_binarize
n_classes = len(le_target.classes_)

for idx, (name, res) in enumerate(results.items()):
    y_proba = res['y_proba']

    # Compute micro-average ROC
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Compute ROC for each class and micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    ax.plot(fpr_micro, tpr_micro,
            color=model_colors[idx], linestyle=line_styles[idx], linewidth=2.5,
            label=f'{name} (AUC = {roc_auc_micro:.3f})')

# Diagonal reference line
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Random Classifier')

ax.set_title('📈 ROC Curves — Micro-Average (All Models)', fontsize=16,
             fontweight='bold', pad=20)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'ml_chart3_roc_curves.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print("   ✅ ML Chart 3: ROC Curves — saved!")

# ════════════════════════════════════════════════════════════
# FIGURE 4: FEATURE IMPORTANCE (Random Forest)
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
rf_model = results['Random Forest']['model']
feature_names = ['Price', 'Quantity', 'Month', 'City', 'Payment Method', 'Day of Week']
importances = rf_model.feature_importances_

# Sort by importance
sorted_idx = np.argsort(importances)
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_imp = importances[sorted_idx]

bars = ax.barh(sorted_names, sorted_imp,
               color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a'],
               edgecolor='white', linewidth=1.5, height=0.5, zorder=3)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{width:.3f}', ha='left', va='center',
            fontweight='bold', fontsize=11)

ax.set_title('🌲 Feature Importance (Random Forest)', fontsize=16,
             fontweight='bold', pad=20)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'ml_chart4_feature_importance.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print("   ✅ ML Chart 4: Feature Importance — saved!")

# ════════════════════════════════════════════════════════════
# FIGURE 5: METRICS COMPARISON (Grouped Bar Chart)
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_names))
width = 0.22

for idx, (name, res) in enumerate(results.items()):
    values = [res['accuracy'], res['precision'], res['recall'], res['f1_score']]
    bars = ax.bar(x + idx * width, [v * 100 for v in values],
                  width, label=name, color=model_colors[idx],
                  edgecolor='white', linewidth=1.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val*100:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9)

ax.set_title('📊 Model Performance Metrics Comparison', fontsize=16,
             fontweight='bold', pad=20)
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, max([res['accuracy'] for res in results.values()]) * 100 + 20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'ml_chart5_metrics_comparison.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print("   ✅ ML Chart 5: Metrics Comparison — saved!")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 10: SAVE RESULTS
# ─────────────────────────────────────────────────────────────

# Save model results to CSV
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1_Score': [results[m]['f1_score'] for m in results]
})
results_df.to_csv(os.path.join(DATA_DIR, 'ml_results.csv'), index=False)
print("\n✅ Model results saved to 'data/ml_results.csv'")

# ─────────────────────────────────────────────────────────────
# 🔹 STEP 11: SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("📋 FINAL ML SUMMARY REPORT")
print("=" * 60)

print(f"""
┌──────────────────────────────────────────────────────────┐
│            MACHINE LEARNING PROJECT SUMMARY               │
├──────────────────────────────────────────────────────────┤
│  Dataset          : E-Commerce Sales (Cleaned)            │
│  Samples          : {len(df)} records                            │
│  Features Used    : {len(feature_cols)} features                          │
│  Target Variable  : Category (Multi-class)                │
│  Train/Test Split : 75% / 25%                             │
├──────────────────────────────────────────────────────────┤
│  MODELS TRAINED:                                          │
│  1. Logistic Regression  → {results['Logistic Regression']['accuracy']*100:5.1f}% accuracy            │
│  2. Decision Tree        → {results['Decision Tree']['accuracy']*100:5.1f}% accuracy            │
│  3. Random Forest        → {results['Random Forest']['accuracy']*100:5.1f}% accuracy            │
├──────────────────────────────────────────────────────────┤
│  🏆 Best Model: {best_model_name:<25}              │
│     Accuracy : {best_acc*100:.1f}%                                    │
├──────────────────────────────────────────────────────────┤
│  Charts Generated : 5 (saved as PNG)                      │
│  Results CSV      : ml_results.csv                        │
└──────────────────────────────────────────────────────────┘
""")

print("✅ ML Project completed successfully!")
print("=" * 60)
