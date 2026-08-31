import os
import requests

api_key = os.getenv("WATSONX_APIKEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
url = os.getenv("WATSONX_URL", "https://eu-de.ml.cloud.ibm.com")

# Get token
token_response = requests.post(
    "https://iam.cloud.ibm.com/identity/token",
    data={
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
)

token = token_response.json()["access_token"]

# Test connection
response = requests.get(
    f"{url}/ml/v1/foundation_model_specs?version=2023-05-29",
    headers={
        "Authorization": f"Bearer {token}"
    }
)

if response.status_code == 200:
    print("✅ Connection successful!")
    print("Available models:", len(response.json().get("resources", [])))
else:
    print(f"❌ Error: {response.status_code}")
    print(response.json())
