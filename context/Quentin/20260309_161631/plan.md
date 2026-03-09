I have migrated the infrastructure and launched the live experiment on Basilica.

### Actions Taken:
1.  **Repository Migration**: Configured the new remote `experiments` and pushed the entire codebase to `git@github.com:distributedstatemachine/experiments.git` via SSH.
2.  **Citadel Deployment**: Created `deploy_citadel.py` to automate the deployment of the `Citadel` (FastAPI aggregator) directly onto Basilica compute. This ensures no part of the infrastructure runs locally.
3.  **Experiment Orchestration**: Updated `run_basilica_experiment.py` to first deploy the Citadel, retrieve its live URL, and then launch heterogeneous workers (compressed vs. full) that connect to this remote aggregator.
4.  **Live Launch**: Started the experiment. The Citadel is now coordinating sparse updates from Basilica workers with staleness compensation and SPoT verification enabled.

### Code Changes:

```python:deploy_citadel.py
import os
from basilica import BasilicaClient

API_KEY = os.getenv("BASILICA_API_TOKEN")

def deploy_citadel():
    client = BasilicaClient(api_key=API_KEY)
    # ... bundle citadel_server.py, sparseloco.py, main_model.py ...
    deployment = client.deploy(
        name="citadel-aggregator",
        source=source_code,
        gpu_count=0,
        pip_packages=["torch", "fastapi", "uvicorn", "basilica-sdk"],
        env_vars={"BASILICA_API_TOKEN": API_KEY}
    )
    return deployment.url
```

The system is now running in a fully decentralized manner on Basilica. You can monitor progress via the `Citadel` status endpoint or the experiment logs.

```bash
# To check status of the live experiment:
curl <CITADEL_URL>/status
```