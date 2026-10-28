
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# The 4 Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Tree Diagram & Evaluation Metrics
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =============================================================================
# PART 1: DATA PREPROCESSING & CLEANING
# =============================================================================

# 1.1 Load Dataset
df = pd.read_csv("C:/Users/chait/Downloads/Bank_customer_churn_Modelling.csv")

print("--- 1. Data Structure & Summary ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- First 5 Rows ---")
print(df.head())

# 1.2 Check Missing Values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# 1.3 Drop Irrelevant Identification Columns (Mam's standard practice)
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# 1.4 Encode Categorical Variables using LabelEncoder (Mam's Loop Method)
cat_cols = ['Geography', 'Gender']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n--- Data Sample After Encoding ---")
print(df.head())


# =============================================================================
# PART 2: OUTLIERS DETECTION (IQR) & EDA VISUALIZATIONS (MAM'S EXACT STYLE)
# =============================================================================
print("\n" + "="*50)
print("PART 2: OUTLIERS DETECTION (IQR METHOD) & EDA")
print("="*50)

# 2.1 Outlier Detection using Interquartile Range (IQR) (Mam's Exact Logic from Unit_4_01)
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
print("Detecting Outliers using IQR Method:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    print(f"  - {col:<16}: {outlier_count} outliers (Bounds: [{lower_bound:.1f}, {upper_bound:.1f}])")

df_normalized = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())

plt.figure(figsize=(11, 5))
sns.boxplot(data=df_normalized, palette='Set2')
plt.title("Visualization 1: Normalized Boxplot for Outlier Detection (Scale: 0 to 1)")
plt.ylabel("Normalized Value (0 to 1)")
plt.xlabel("Numerical Features")
plt.tight_layout()
plt.show()

# 2.3 Visualization 2: Histogram of Customer Age (Mam's Purple Bars & Black Edge)
plt.figure(figsize=(8, 4))
plt.hist(df['Age'], bins=30, color='purple', edgecolor='black', alpha=0.8)
plt.title("Visualization 2: Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2.4 Visualization 3: Countplot - Customer Churn across Geography (Mam's Countplot)
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='Geography', hue='Exited', palette='coolwarm')
plt.title("Visualization 3: Customer Churn Distribution by Geography")
plt.xlabel("Geography (0: France, 1: Germany, 2: Spain)")
plt.ylabel("Number of Customers")
plt.legend(title='Exited', labels=['Stayed (0)', 'Exited (1)'])
plt.tight_layout()
plt.show()

# 2.5 Visualization 4: Scatter Plot - Age vs Balance (Mam's Scatter Plot)
plt.figure(figsize=(9, 5))
sns.scatterplot(x='Age', y='Balance', hue='Exited', data=df, alpha=0.7, palette='coolwarm')
plt.title("Visualization 4: Scatter Plot of Age vs Balance (Colored by Churn)")
plt.xlabel("Age")
plt.ylabel("Account Balance")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2.6 Visualization 5: Correlation Heatmap (Mam's Heatmap)
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Visualization 5: Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# =============================================================================
# PART 3: DATA SPLITTING & FEATURE SCALING
# =============================================================================

# Feature matrix X and Target y
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train Test Split (test_size = 0.20, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# StandardScaler (fit_transform on train, transform on test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# PART 4: MODEL 1 - LOGISTIC REGRESSION (Regression Baseline)
# =============================================================================
print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed (0)', 'Exited (1)'], 
            yticklabels=['Stayed (0)', 'Exited (1)'])
plt.title("Model 1: Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# =============================================================================
# PART 5: MODEL 2 - K-NEAREST NEIGHBORS (KNN / Lazy Learning)
# =============================================================================
print("\n" + "="*50)
print("MODEL 2: K-NEAREST NEIGHBORS (KNN)")
print("="*50)

knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Greens',
            xticklabels=['Stayed (0)', 'Exited (1)'], 
            yticklabels=['Stayed (0)', 'Exited (1)'])
plt.title("Model 2: KNN (k=5) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# =============================================================================
# PART 6: MODEL 3 - GAUSSIAN NAIVE BAYES (Probabilistic Learning)
# =============================================================================
print("\n" + "="*50)
print("MODEL 3: GAUSSIAN NAIVE BAYES")
print("="*50)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

# Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Stayed (0)', 'Exited (1)'], 
            yticklabels=['Stayed (0)', 'Exited (1)'])
plt.title("Model 3: Gaussian Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# =============================================================================
# PART 7: MODEL 4 - RANDOM FOREST CLASSIFIER (Mam's Exact Screenshot Code)
# =============================================================================
print("\n" + "="*50)
print("MODEL 4: RANDOM FOREST CLASSIFIER")
print("="*50)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_rf)
acc = accuracy_score(y_test, y_pred_rf)

print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Stayed (0)', 'Exited (1)'], 
            yticklabels=['Stayed (0)', 'Exited (1)'])
plt.title("Model 4: Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Mam's Exact Random Forest Diagram (First Decision Tree Plot)
# -----------------------------------------------------------------------------
tree = rf.estimators_[0]
plt.figure(figsize=(22, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["Stayed", "Exited"],
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=9
)

plt.title("Random Forest - First Decision Tree")
plt.tight_layout()
plt.show()


# =============================================================================
# PART 8: FINAL SUMMARY PERFORMANCE COMPARISON TABLE
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: FINAL PERFORMANCE COMPARISON ACROSS ALL 4 MODELS")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': [
        'Logistic Regression', 
        'K-Nearest Neighbors (k=5)', 
        'Gaussian Naive Bayes', 
        'Random Forest (100 Trees)'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_rf)
    ]
})

print(comparison_df.to_string(index=False))