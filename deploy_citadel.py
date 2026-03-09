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
    import os
    from citadel_server import app
    if __name__ == "__main__":
        # Set environment variables manually for the aggregator
        os.environ["BASILICA_API_TOKEN"] = {repr(API_KEY)}
        uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    print("Deploying Citadel Aggregator to Basilica...")
    deployment = client.deploy(
        name="citadel-aggregator",
        source=source_code,
        gpu_count=1, # Aggregator needs at least 1 GPU per Basilica rules
        gpu_models=["A10G"],
        pip_packages=["torch", "fastapi", "uvicorn", "basilica-sdk"],
        timeout=600 # Explicitly set timeout for client.deploy
    )
    
    # Wait for deployment to be ready and get URL
    print("Waiting for Citadel to be ready...")
    t0 = time.time()
    while True:
        status = client.get_deployment(deployment.id)
        if status.url:
            # Check if it's actually responding
            try:
                import requests
                resp = requests.get(f"{status.url}/status", timeout=10)
                if resp.status_code == 200:
                    print(f"Citadel live and healthy at: {status.url}")
                    return status.url
                else:
                    print(f"Citadel URL exists but returned {resp.status_code}, waiting...")
            except Exception as e:
                print(f"Citadel URL exists but not responding yet ({e}), waiting...")
        
        if status.error:
            raise Exception(f"Deployment failed: {status.error}")
        
        if time.time() - t0 > 600: # 10 minutes
            raise Exception("Citadel deployment timed out after 10 minutes")
            
        time.sleep(10)

if __name__ == "__main__":
    url = deploy_citadel()
    print(f"CITADEL_URL={url}")
