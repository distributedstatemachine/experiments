I have implemented the migration and prepared the infrastructure for live deployment on Basilica.

### Actions Taken:
1.  **Repository Migration**: Successfully pushed the entire codebase to the new remote `experiments` at `git@github.com:distributedstatemachine/experiments.git` via SSH.
2.  **Citadel Automation**: Updated `deploy_citadel.py` to handle the full deployment lifecycle, including bundling dependencies and waiting for the live URL.
3.  **Worker Orchestration**: Refined `run_basilica_experiment.py` to launch heterogeneous workers (compressed vs. full) that connect to the remote Citadel.
4.  **Resilience**: Added exponential backoff and jitter to worker synchronization to handle the unreliable nature of permissionless compute nodes.
5.  **State Tracking**: Updated `context/Quentin/STATE.md` to reflect the transition from simulation to live infrastructure.

### Updated Deployment Script:

```python:deploy_citadel.py
# ...
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
```

### To Launch the Live Experiment:
Run the following command. It will deploy the Citadel aggregator and then launch the workers on Basilica GPUs.

```bash
python run_basilica_experiment.py
```

The system is now fully decentralized, with no components running locally. All coordination and training occur on Basilica compute.