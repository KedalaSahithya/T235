from huggingface_hub import HfApi, Repository
import os
import shutil
import sys

def upload_to_huggingface(repo_name, username=None):
    """
    Upload the water forecasting model and data to Hugging Face Hub.
    
    Args:
        repo_name (str): Name of the repository to create on Hugging Face Hub
        username (str, optional): Your Hugging Face username. If None, will use the value from the HF_USERNAME env variable.
    """
    if username is None:
        username = os.environ.get("HF_USERNAME")
        if username is None:
            print("Error: Please provide a username or set the HF_USERNAME environment variable.")
            sys.exit(1)
    
    # Full repository name
    full_repo_name = f"{username}/{repo_name}"
    
    # Check if token is available
    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("Error: HF_TOKEN environment variable not set. Please set it with your Hugging Face API token.")
        print("You can get your token from https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create a temporary directory for the repository
    repo_dir = "hf_repo"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    os.makedirs(repo_dir)
    
    try:
        # Create or clone the repository
        print(f"Creating repository: {full_repo_name}")
        try:
            repo = Repository(
                local_dir=repo_dir,
                clone_from=full_repo_name,
                use_auth_token=token
            )
        except Exception:
            # Repository doesn't exist yet, create it
            api.create_repo(
                repo_id=repo_name,
                private=False,
                exist_ok=True
            )
            repo = Repository(
                local_dir=repo_dir,
                clone_from=full_repo_name,
                use_auth_token=token
            )
        
        # Create directories in the repository
        os.makedirs(os.path.join(repo_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(repo_dir, "data"), exist_ok=True)
        
        # Copy model and data files to the repository
        print("Copying model and data files...")
        for file in os.listdir("models"):
            shutil.copy(os.path.join("models", file), os.path.join(repo_dir, "models", file))
        
        for file in os.listdir("data"):
            shutil.copy(os.path.join("data", file), os.path.join(repo_dir, "data", file))
        
        # Create a README.md file
        with open(os.path.join(repo_dir, "README.md"), "w") as f:
            f.write(f"""# Water Forecasting Model

This repository contains a machine learning model for forecasting water demand and assessing reservoir storage capacities.

## Model Description

The model uses Random Forest Regression to predict future water requirements based on:
- Population growth
- Rainfall patterns
- Temperature trends
- Historical water usage

## Files

- `models/water_demand_model.joblib`: Trained Random Forest model for water demand forecasting
- `data/water_data.csv`: Historical water data
- `data/reservoir_info.csv`: Information about reservoir capacities
- `data/feature_importance.csv`: Feature importance from the model
- `data/scenario_*.csv`: Example scenarios for forecasting

## Usage

This model can be used with the companion Gradio web app available at: https://huggingface.co/spaces/{username}/water-forecasting-app

## Citation

If you use this model in your research or project, please cite: