I've refactored the codebase to explicitly separate the `Quentin` (Decentralized Training Research) and `Quant` (Trading System) tasks. This ensures that the two agents operate in distinct environments while sharing core infrastructure.

### Changes Made:

1.  **Directory Isolation**:
    *   `quentin/`: Contains all core research on SparseLoCo, Basilica integration, SPoT/ZK-SPoT, and decentralized orchestration.
    *   `quant/`: Contains trading-specific logic, including `TimeSeriesFoundationModel`, Hyperliquid data ingestion, and backtesting.
    *   `shared/`: Holds common models (like `GlobalModel`) used by both systems.

2.  **Code Refactoring**:
    *   Updated all imports in `quentin/` and `quant/` to use the new namespaced structure.
    *   Refined `quentin/deploy_citadel.py` and `quentin/run_basilica_experiment.py` to correctly bundle files from their new locations during worker deployment.
    *   Fixed `quant/quant_worker.py` and `quant/quant_aggregator.py` to point to the correct research modules in `quentin/`.

3.  **State Management**:
    *   Updated `context/Quentin/STATE.md` and `context/quant/STATE.md` to reflect the new structure and distinct research goals.
    *   The `quant` agent is now clearly defined as building a trading system *using* the algorithms developed by `Quentin`.

4.  **Git**: Committed all changes to the repository.

```bash
# New Structure
.
├── quant/                  # Trading System (TimeSeries, Hyperliquid)
├── quentin/                # Decentralized Training (SparseLoCo, SPoT)
├── shared/                 # Shared Model Definitions
└── context/                # Agent-specific state and logs
```

The codebase is now clean, and the two agents have clear operational boundaries. `Quentin` can continue looping on convergence/performance, while `Quant` focuses on financial applications.