from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="wellness_tourism_package_customer_prediction_dummy/model_deployment",
    repo_id="haseena84/Wellness-Tourism-Package-Customer-Prediction-dummy",
    repo_type="space",
    path_in_repo="",
)
