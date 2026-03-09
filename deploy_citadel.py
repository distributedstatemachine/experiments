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
    
    print(f"Citadel live at: {deployment.url}")
    return deployment.url

if __name__ == "__main__":
    url = deploy_citadel()
    print(f"CITADEL_URL={url}")
