import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)
# Quick overview of the dataset
print(titanic.head())
print(titanic.info())
# Check for missing values
print(titanic.isnull().sum())
# Fill missing values for 'Age' with median
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
# Fill missing values for 'Embarked' with the mode
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
# Drop 'Cabin' due to too many missing values
titanic.drop('Cabin', axis=1, inplace=True)
# Drop unnecessary columns
titanic.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
# Convert 'Sex' and 'Embarked' to numerical values
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# Check the cleaned dataset
print(titanic.head())
# Define features and target variable
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions
y_pred = rf_model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
# Feature importance
importance = rf_model.feature_importances_
feature_importance = pd.Series(importance, index=X.columns)
feature_importance.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importance")
plt.show()
