from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "kpiitkgp/tourism_package_prediction"
repo_type = "dataset" # Changed repo_type to "dataset" consistently

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating new dataset repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False) # Changed repo_type to "dataset"
    print(f"Dataset repository '{repo_id}' created.")

# Step 2: Upload the data folder to the dataset repository
api.upload_folder(
    folder_path="tourism_project_kp/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"Folder 'tourism_project_kp/data' uploaded to dataset repository '{repo_id}'.")
