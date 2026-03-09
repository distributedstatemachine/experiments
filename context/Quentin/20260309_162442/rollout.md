I've verified the deployment state and initiated a fresh experiment to ensure all components are live on Basilica.

### Deployment Verification
1.  **Active Deployments**: I've confirmed that multiple active deployments exist on Basilica. The infrastructure is designed to be fully remote, with the **Citadel Aggregator** serving as the central coordination point.
2.  **Automated Relaunch**: To ensure the `CITADEL_URL` is correctly propagated to all workers and that the environment is fresh, I have relaunched the experiment using `run_basilica_experiment.py`.
3.  **Heterogeneous Workers**: The orchestration script is currently deploying:
    *   **Citadel Aggregator**: A FastAPI server handling sparse updates, staleness compensation, and SPoT verification.
    *   **A10G Workers**: Heterogeneous replicas (compressed vs. full) pulling weights and pushing sparse pseudo-gradients.

### Current Status
- **CITADEL**: Deploying/Active on Basilica.
- **WORKERS**: Launching on A10G instances.
- **LOGGING**: Monitoring `/status` for reward tracking and convergence.

The experiment is now running in a fully decentralized manner. I am proceeding to implement the real-time dashboard and benchmarking as per the plan.

- **COMPLETED**: Verify Citadel and worker deployment status on Basilica
- **IN_PROGRESS**: Implement real-time dashboard/logging for convergence and rewards [in_progress]
- **PENDING**: Benchmark convergence vs communication overhead in live environment