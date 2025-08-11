import os
import time
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "BTC_LSTM_Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production") # Or 'Staging' or specific version

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient() # Initialize client once

def wait_for_model(model_name, model_alias, max_retries=60, retry_interval=5):
    print(f"Waiting for model '{model_name}' with alias '{model_alias}' to be available...")
    for i in range(max_retries):
        try:
            # First, check if the alias exists in the registry
            model_version_info = client.get_model_version_by_alias(name=model_name, alias=model_alias)
            model_version = model_version_info.version
            model_source_uri = model_version_info.source # Get the artifact source URI
            print(f"Alias '{model_alias}' found for model '{model_name}', pointing to version {model_version} with source: {model_source_uri}.")

            # Now, attempt to load the model using its direct artifact source URI
            mlflow.pyfunc.load_model(model_source_uri) # Load directly from artifact source
            print(f"Model '{model_name}' (version {model_version}) found and loaded successfully from source URI!")
            return True
        except MlflowException as e:
            # Specific MLflow exceptions for not found
            if "No versions of model with name" in str(e) or "Alias" in str(e) and "not found" in str(e) or "INVALID_PARAMETER_VALUE" in str(e):
                print(f"Attempt {i+1}/{max_retries}: Model alias '{model_alias}' not yet registered or invalid parameter. Retrying...")
            else:
                print(f"Attempt {i+1}/{max_retries}: Error loading model: {e}. Retrying...")
            time.sleep(retry_interval)
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries}: An unexpected error occurred: {e}. Retrying...")
            time.sleep(retry_interval)
    print(f"Max retries reached. Model '{model_name}' with alias '{model_alias}' not found or could not be loaded.")
    return False

if __name__ == "__main__":
    if not wait_for_model(REGISTERED_MODEL_NAME, MODEL_ALIAS):
        print("Failed to find model after multiple retries. Exiting.")
        exit(1)

    print("Starting Uvicorn server...")
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])