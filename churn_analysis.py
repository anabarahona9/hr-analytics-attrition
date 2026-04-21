import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv('D:/data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Exploratory Analysis
print("Shape", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:", df.dtypes)
print("\nNull values:", df.isnull().sum())
print("\nTarget variable distribution", df["Attrition"].value_counts())
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
print(df.shape)

# Visual EDA
# Attrition by Department
plt.figure(figsize = (10,5))
sns.countplot(data=df, x="Department", hue="Attrition")
plt.title("Attrition by Department")
plt.savefig("attrition_department.png")

# Attrition by Age
plt.figure(figsize = (10,5))
sns.histplot(data = df, x="Age", hue="Attrition", bins = 20)
plt.title("Attrition by Age")
plt.savefig("attrition_age.png")

# Monthly income vs Attrition
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
plt.title("Monthly Income vs Attrition")
plt.savefig("attrition_income.png")

# Correlation Heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.select_dtypes(include="int64").corr(),
            annot=True, fmt='.1f', cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

# Convert categorical columns to numbers
le = LabelEncoder()
categorical_cols = df.select_dtypes(include= 'str').columns.tolist()
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
print(df.dtypes)

# Define features and target
X= df.drop(columns=['Attrition'])
Y= df['Attrition']

#Split 80% train, 20% test

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

# Model: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred_rf = rf_model.predict(X_test)

print("==Random Forest==")
print(classification_report(Y_test, Y_pred_rf))
print(confusion_matrix(Y_test, Y_pred_rf))

# Balance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, Y_train)

print("Before SMOTE:", Y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_balanced).value_counts().to_dict())

# Model 2: Random Forest with SMOTE
rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model2.fit(X_train_balanced, y_train_balanced)
y_pred_rf2 = rf_model2.predict(X_test)

print("=== RANDOM FOREST + SMOTE ===")
print(classification_report(Y_test, y_pred_rf2))

# Feature Importance with Random Forest
importances = pd.Series(rf_model2.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Top 10 Features - Employee Attrition')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Export to csv
df_original = pd.read_csv('D:/data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df_original['Attrition_Binary'] = df['Attrition']
df_original['Attrition_Predicted'] = rf_model2.predict(df.drop(columns=['Attrition']))
df_original.to_csv('D:/data/churn_results.csv', index=False)

