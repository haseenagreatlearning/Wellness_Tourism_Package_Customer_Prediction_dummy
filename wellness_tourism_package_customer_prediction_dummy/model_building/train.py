
# Importing Necessary Libraries
import os
import joblib
import pandas as pd
import mlflow

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


DATASET_REPO_ID = "haseena84/Wellness-Tourism-Package-Customer-Prediction-dummy"
MODEL_REPO_ID = "haseena84/Wellness-Tourism-Package-Customer-Prediction-dummy"
REPO_TYPE_MODEL = "model"
TARGET_COL = "ProdTaken"

Xtrain_path = "hf://datasets/haseena84/Wellness-Tourism-Package-Customer-Prediction-dummy/Xtrain.csv"
Xtest_path  = "hf://datasets/haseena84/Wellness-Tourism-Package-Customer-Prediction-dummy/Xtest.csv"
ytrain_path = "hf://datasets/haseena84/Wellness-Tourism-Package-Customer-Prediction-dummy/ytrain.csv"
ytest_path  = "hf://datasets/haseena84/Wellness-Tourism-Package-Customer-Prediction-dummy/ytest.csv"


# Load Data

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)[TARGET_COL]
ytest  = pd.read_csv(ytest_path)[TARGET_COL]

numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

# Class Weight to Handle Imbalance
class_counts = ytrain.value_counts()
if 0 in class_counts and 1 in class_counts and class_counts[1] != 0:
    class_weight = class_counts[0] / class_counts[1]
else:
    class_weight = 1.0

# Preprocessing

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="drop"
)

# Define Model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=float(class_weight),
    random_state=42,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1
)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__subsample": [0.6, 0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))


with mlflow.start_run():
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("scale_pos_weight", float(class_weight))
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("encoder", "OneHotEncoder")
    mlflow.log_param("scoring", "recall")
    mlflow.log_param("grid_candidates", 243)

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring="recall", n_jobs=-1)

    grid_search.fit(Xtrain, ytrain)


    mlflow.log_metric("best_cv_recall", float(grid_search.best_score_))
    mlflow.log_params(grid_search.best_params_)


    cv_path = "cv_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(cv_path, index=False)
    mlflow.log_artifact(cv_path, artifact_path="gridsearch")

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    print("\nTraining Classification Report:")
    print(classification_report(ytrain, y_pred_train))

    print("\nTest Classification Report:")
    print(classification_report(ytest, y_pred_test))

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy  = accuracy_score(ytest, y_pred_test)
    train_recall   = recall_score(ytrain, y_pred_train)
    test_recall    = recall_score(ytest, y_pred_test)

    mlflow.log_metrics({
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_recall": float(train_recall),
        "test_recall": float(test_recall),
        "train_precision_pos1": float(train_report["1"]["precision"]),
        "train_f1_pos1": float(train_report["1"]["f1-score"]),
        "test_precision_pos1": float(test_report["1"]["precision"]),
        "test_f1_pos1": float(test_report["1"]["f1-score"]),
    })

    # Save model
    model_filename = "best_tourist_customer_xgb_model.joblib"
    joblib.dump(best_model, model_filename)
    mlflow.log_artifact(model_filename, artifact_path="model")



# Upload model to Hugging Face Hub

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set. Please configure it in your environment / GitHub Secrets.")

api = HfApi(token=hf_token)

try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE_MODEL)
    print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{MODEL_REPO_ID}' not found. Creating a new one...")
    create_repo(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE_MODEL, private=False)
    print(f"Model repo '{MODEL_REPO_ID}' created successfully.")

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=MODEL_REPO_ID,
    repo_type=REPO_TYPE_MODEL,
)
print(f"Uploaded model to HF: {MODEL_REPO_ID}/{model_filename}")
