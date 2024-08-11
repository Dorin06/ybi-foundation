# Title of Project: Bank Customer Churn Prediction Model

# Objective:
# To predict whether a customer will churn based on historical customer data.

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import Data
data = pd.read_csv('Bank_Customer_Churn_Dataset.csv')

# Describe Data
print("First few rows of the dataset:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Data Visualization
# Plot the distribution of the target variable (Churn)
sns.countplot(x='Churn', data=data)
plt.title('Distribution of Churn')
plt.show()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of Age
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

# Boxplot for Balance by Churn
sns.boxplot(x='Churn', y='Balance', data=data)
plt.title('Boxplot of Balance by Churn')
plt.show()

# Data Preprocessing
# Handling Missing Values
print("\nMissing values in the dataset:")
print(data.isnull().sum())
data = data.dropna()  # Drop missing values (if any)

# Encoding Categorical Variables
le = LabelEncoder()
data['Geography'] = le.fit_transform(data['Geography'])
data['Gender'] = le.fit_transform(data['Gender'])

# Feature Scaling
scaler = StandardScaler()
features_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Define Target Variable (y) and Feature Variables (X)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print(f'\nModel Accuracy: {accuracy}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)
