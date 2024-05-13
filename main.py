import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pickle
import os

# Load dataset
try:
    df = pd.read_csv('creditcard.csv')  # Ensure correct path to the CSV file
except FileNotFoundError:
    print("Error: The specified file could not be found. Please check the path.")
    df = None

# Data Exploration Function
def explore_data(df):

    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns

    if numeric_df.empty:
        print("No data to explore.")
        return

    # Summary statistics
    print("Summary statistics:")
    print(df.describe())  # If nothing is displayed, check if the dataframe has data

    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())  # Ensure that this prints some information

    # Display sample data
    print("Sample data:")
    print(df.head())  # This should print the first few rows of the dataframe

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# Run the exploration function
if df is not None:
    explore_data(df)

# Function for data preprocessing with success/error reporting
def preprocess_data(df, features, categorical_columns=None, scale=True):
    success = False
    result_message = ""
    preprocessor = None
    X_encoded = None
    Y = None

    try:
        
        # Select features and target
        X = df[features]
        Y = df['Class']
        
        
        # Identify numeric columns
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        
        # Convert numeric columns to float32
        X.loc[:, numeric_cols] = X.loc[:, numeric_cols].astype('float32')
        
        # Convert other columns to numeric and handle errors
        for col in X.columns:
            if col not in numeric_cols:
                X.loc[:, col] = pd.to_numeric(X.loc[:, col], errors='coerce')
        
        # Create transformers
        transformers = []
        if categorical_columns:
            transformers.append(('cat', OneHotEncoder(), categorical_columns))
        if scale:
            transformers.append(('num', StandardScaler(), numeric_cols))
        
        # Create ColumnTransformer
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        
        # Fit and transform the data
        X_encoded = preprocessor.fit_transform(X)
        success = True
        result_message = "Data preprocessing successful."
    
    except Exception as e:
        result_message = f"Error during data preprocessing: {e}"
    
    return X_encoded, Y, preprocessor, success, result_message
    

# Function to evaluate the model with success/error reporting
def evaluate_model(model, X_train, X_test, Y_train, Y_test, threshold=0.5, use_case=None):
    success = False
    result_message = ""
    metrics = {}

    try:
        # Ensure model is initialized
        assert model is not None, "Model is not initialized."
        
        # Make predictions
        train_predictions = (model.predict_proba(X_train)[:, 1] >= threshold).astype(int)
        test_predictions = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        
        # Display classification reports
        print(f"Training Report:\n{classification_report(Y_train, train_predictions)}")
        print(f"Testing Report:\n{classification_report(Y_test, test_predictions)}")
        
        # Confusion matrix
        cm = confusion_matrix(Y_test, test_predictions)
        assert cm is not None, "Confusion matrix generation failed."
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Use Case {use_case}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(Y_test, test_predictions),
            'precision': precision_score(Y_test, test_predictions, zero_division=0),
            'recall': recall_score(Y_test, test_predictions),
            'f1_score': f1_score(Y_test, test_predictions)
        }
        
        success = True
        result_message = "Model evaluation successful."

    except Exception as e:
        result_message = f"Error during model evaluation: {e}"

    return metrics, success, result_message


# Function to apply SMOTE with success/error reporting
def apply_smote(X_train, Y_train):
    """
    Validate SMOTE application to balance classes.
    """
    smote = SMOTE(random_state=42)
    try:
        X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)
        # Ensure the lengths match to avoid mismatch issues
        if sp.issparse(X_resampled):
            assert X_resampled.shape[0] == len(Y_resampled), "Mismatch in resampled data."
        else:
            assert len(X_resampled) == len(Y_resampled), "Mismatch in resampled data."
        return X_resampled, Y_resampled
    except AssertionError as ae:
        raise RuntimeError(f"Error: {ae}")
    except Exception as e:
        raise RuntimeError(f"Error during SMOTE application: {e}")
    
# Define functions for model training and evaluation for each use case
def use_case_1(df):
    # Define features for preprocessing
    features = ['State', 'City', 'Street', 'zip_code']
    categorical_columns = ['State', 'City', 'Street','zip_code']

    # Call preprocess_data and unpack the results
    X_encoded, Y, preprocessor, success, result_message = preprocess_data(df, features, categorical_columns)

    # Check if preprocessing was successful
    if not success:
        raise RuntimeError(f"Preprocessing failed: {result_message}")

    # Continue with other processing if successful
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, stratify=Y, random_state=42)

    # Apply SMOTE to training data
    X_train_resampled, Y_train_resampled = apply_smote(X_train, Y_train)

    # Train model (example: logistic regression)
    model = LogisticRegression(solver='saga', max_iter=500, penalty='l2', C=1.0)
    model.fit(X_train_resampled, Y_train_resampled)

    # Evaluate the model
    metrics = evaluate_model(model, X_train, X_test, Y_train, Y_test, use_case=1)

    return model, preprocessor, metrics


def use_case_2(df):
    # Features for Use Case 2
    features = ['MerchantCategory', 'MerchantName', 'Amount', 'Frequency','first','last']
    categorical_columns = ['MerchantCategory', 'MerchantName','first','last']
    
    # Call preprocess_data and unpack the results
    X_encoded, Y, preprocessor, success, result_message = preprocess_data(df, features, categorical_columns)

    # Check if preprocessing was successful
    if not success:
        raise RuntimeError(f"Preprocessing failed: {result_message}")

    # Continue with other processing if successful
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Apply SMOTE to the training data
    X_train, Y_train = apply_smote(X_train, Y_train)
    
    # Train the model (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, Y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_train, X_test, Y_train, Y_test,use_case=2)
    
    return model, preprocessor, metrics

def use_case_3(df):
    # Features for Use Case 3
    features = ['Amount', 'AverageTransactionAmount','unix_time']
    
    # Call preprocess_data and unpack the results
    X_encoded, Y, preprocessor, success, result_message = preprocess_data(df, features)

    # Check if preprocessing was successful
    if not success:
        raise RuntimeError(f"Preprocessing failed: {result_message}")

    # Continue with other processing if successful
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Apply SMOTE to the training data
    X_train, Y_train = apply_smote(X_train, Y_train)
    
    # Train the model (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, Y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_train, X_test, Y_train, Y_test,use_case=3)
    
    return model, preprocessor, metrics

def use_case_4(df):
    # Features for Use Case 4
    features = ['Amount', 'FrequencyOfTransactionWithSameAmount']
    
    # Call preprocess_data and unpack the results
    X_encoded, Y, preprocessor, success, result_message = preprocess_data(df, features)

    # Check if preprocessing was successful
    if not success:
        raise RuntimeError(f"Preprocessing failed: {result_message}")

    # Continue with other processing if successful
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Apply SMOTE to the training data
    X_train, Y_train = apply_smote(X_train, Y_train)
    
    # Train the model (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, Y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_train, X_test, Y_train, Y_test,use_case=4)
    
    return model, preprocessor, metrics

# Run each use case
model_uc1, preprocessor_uc1, metrics_uc1 = use_case_1(df)
model_uc2, preprocessor_uc2, metrics_uc2 = use_case_2(df)
model_uc3, preprocessor_uc3, metrics_uc3 = use_case_3(df)
model_uc4, preprocessor_uc4, metrics_uc4 = use_case_4(df)

print('Successfully completed all use cases.')

# Directory where the models will be stored
models_directory = "models"
os.makedirs(models_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Save the models and preprocessors for each use case
with open(os.path.join(models_directory, "uc1_model.pickle"), "wb") as f:
    pickle.dump(model_uc1, f)

with open(os.path.join(models_directory, "uc1_preprocessor.pickle"), "wb") as f:
    pickle.dump(preprocessor_uc1, f)

with open(os.path.join(models_directory, "uc2_model.pickle"), "wb") as f:
    pickle.dump(model_uc2, f)

with open(os.path.join(models_directory, "uc2_preprocessor.pickle"), "wb") as f:
    pickle.dump(preprocessor_uc2, f)

with open(os.path.join(models_directory, "uc3_model.pickle"), "wb") as f:
    pickle.dump(model_uc3, f)

with open(os.path.join(models_directory, "uc3_preprocessor.pickle"), "wb") as f:
    pickle.dump(preprocessor_uc3, f)

with open(os.path.join(models_directory, "uc4_model.pickle"), "wb") as f:
    pickle.dump(model_uc4, f)

with open(os.path.join(models_directory, "uc4_preprocessor.pickle"), "wb") as f:
    pickle.dump(preprocessor_uc4, f)

