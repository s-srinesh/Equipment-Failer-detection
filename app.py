from flask import Flask, request, render_template
from flask_cors import CORS
import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React frontend

# Load dataset (replace with actual file path)
df = pd.read_csv('equipment_failure_dataset.csv')

# Preprocessing steps
df = df.drop(columns=['EquipmentID'])  # Drop EquipmentID as it's not useful for prediction
X = df.drop(columns=['Failure'])
y = df['Failure'].apply(lambda x: 1 if x == 'Yes' else 0)

# Create preprocessing pipelines
numerical_cols = ['Age', 'UsageHours', 'MaintenanceHistory', 'Temperature', 'Pressure', 'VibrationLevel', 'OperatorExperience', 'FailureHistory']
categorical_cols = ['Location', 'Environment']

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models for Voting Classifier
models = [
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('lr', LogisticRegression(max_iter=1000)),
    ('svc', SVC(probability=True)),
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier()),
    ('xgb', XGBClassifier())
]

# Voting Classifier
voting_clf = VotingClassifier(estimators=models, voting='soft')

# Create pipeline with preprocessor and model
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

# Train the model within the pipeline
clf_pipeline.fit(X_train, y_train)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [
            float(data['age']),
            float(data['usageHours']),
            float(data['maintenanceHistory']),
            float(data['temperature']),
            float(data['pressure']),
            float(data['vibrationLevel']),
            float(data['operatorExperience']),
            float(data['failureHistory']),
            data['location'],
            data['environment']
        ]
    except (KeyError, ValueError) as e:
        logging.error(f'Input error: {e}')
        return "Invalid input data", 400

    # Create a DataFrame for the features
    feature_names = numerical_cols + categorical_cols
    features_df = pd.DataFrame([features], columns=feature_names)

    try:
        prediction = clf_pipeline.predict(features_df)
    except Exception as e:
        logging.error(f'Model prediction error: {e}')
        return "Prediction failed", 500

    result = 'Equipment fails' if prediction[0] == 1 else 'Equipment does not fail'
    return result, 200  # Plain text response

# Route to render the HTML template
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use the PORT environment variable, defaulting to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
