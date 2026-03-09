# State: Quant Agent - SOTA Trading System

## Current Focus
Building a SOTA quant trading system on Hyperliquid using time-series foundation models (Lag-Llama, Chronos, TimesFM) trained via decentralized compute.

## Research & Analysis
- **Hyperliquid SDK**: Integrated for L2/L3 data ingestion and order execution.
- **Time-Series Foundation Models**: Researching Lag-Llama, Chronos, and TimesFM for zero-shot forecasting.
- **Decentralized Fine-tuning**: Adapting SparseLoCo (from Quentin's research) for financial time-series data.

## Implementation Progress
- [x] Initialized `quant/` directory for trading-specific logic.
- [x] Implemented `quant/quant_model.py` with `TimeSeriesFoundationModel`.
- [x] Implemented `quant/data_ingestion.py` for Hyperliquid L2/L3 data sharding.
- [x] Implemented `quant/quant_worker.py` and `quant/quant_aggregator.py`.
- [x] Implemented `quant/backtest_engine.py` for Hyperliquid historical data.
- [x] Integrated **LAMB** and **Lookahead Aggregation** for decentralized convergence.
- [x] Hardened `quant_worker.py` with **FedSAM** and **Subspace Compression**.
- [x] Implemented **SPoT** and **Coordinate-wise Median** filtering.

## Next Steps
- [ ] Finalize Hyperliquid L2/L3 data ingestion pipeline.
- [ ] Integrate quant model with `quentin/basilica_training.py` for decentralized fine-tuning.
- [ ] Run large-scale pretraining on Basilica with 12+ workers using Quentin's research.
- [ ] Implement ZK-SPoT for privacy-preserving verification of proprietary trading features.
- [x] Refactored codebase to explicitly separate `Quentin` (Research) and `Quant` (Trading) tasks.
- [x] Updated all imports in `quentin/` and `quant/` to use the new namespaced structure.
- [x] Refined `quentin/deploy_citadel.py` and `quentin/run_basilica_experiment.py` to correctly bundle files from their new locations.
- [x] Fixed `quant/quant_worker.py` and `quant/quant_aggregator.py` to point to the correct research modules in `quentin/`.
