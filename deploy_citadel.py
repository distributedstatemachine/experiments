import os
import time
from basilica import BasilicaClient

# Basilica API Key from environment
API_KEY = os.getenv("BASILICA_API_TOKEN")

def deploy_citadel():
    client = BasilicaClient(api_key=API_KEY)
    
    # Read the required files to bundle them
    with open("citadel_server.py", "r") as f:
        citadel_server_code = f.read()
    with open("sparseloco.py", "r") as f:
        sparseloco_code = f.read()
    with open("main_model.py", "r") as f:
        main_model_code = f.read()

    # Bundle the code
    source_code = f"""
import os
import sys

# Write bundled modules to disk
with open("citadel_server.py", "w") as f:
    f.write({repr(citadel_server_code)})
with open("sparseloco.py", "w") as f:
    f.write({repr(sparseloco_code)})
with open("main_model.py", "w") as f:
    f.write({repr(main_model_code)})

# Run the server
import uvicorn
from citadel_server import app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    print("Deploying Citadel Aggregator to Basilica...")
    deployment = client.deploy(
        name="citadel-aggregator",
        source=source_code,
        gpu_count=0, # Aggregator doesn't need GPU
        pip_packages=["torch", "fastapi", "uvicorn", "basilica-sdk"],
        env_vars={
            "BASILICA_API_TOKEN": API_KEY
        }
    )
    
    # Wait for deployment to be ready and get URL
    print("Waiting for Citadel to be ready...")
    while True:
        status = client.get_deployment(deployment.id)
        if status.url:
            print(f"Citadel live at: {status.url}")
            return status.url
        if status.error:
            raise Exception(f"Deployment failed: {status.error}")
        time.sleep(5)

if __name__ == "__main__":
    url = deploy_citadel()
    print(f"CITADEL_URL={url}")
