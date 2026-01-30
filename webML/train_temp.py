import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# Load dataset
try:
    df = pd.read_csv('../scikit-learn/data/titanic/train.csv')
except FileNotFoundError:
    # Try absolute path if relative fails
    df = pd.read_csv('c:/Users/ailee/github/Datasience/scikit-learn/data/titanic/train.csv')

# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = df[features]
y = df[target]

# Define numerical and categorical features
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Numeric Transformer: Impute median, Scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Transformer: Impute frequent, OneHotEncode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

clf1 = LogisticRegression(random_state=42)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = RandomForestClassifier(random_state=42)
clf4 = SVC(probability=True, random_state=42)
clf5 = KNeighborsClassifier()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1), 
        ('dt', clf2), 
        ('rf', clf3), 
        ('svc', clf4), 
        ('knn', clf5)
    ],
    voting='soft'
)

# Create full pipeline with preprocessor and model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', voting_clf)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {accuracy:.4f}")

joblib.dump(model_pipeline, 'titanic_voting_model.pkl')
print("Model saved as titanic_voting_model.pkl")
