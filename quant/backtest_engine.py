import torch
import torch.nn as nn
from quant.quant_model import TimeSeriesFoundationModel
from quant.data_ingestion import HyperliquidDataIngestor
import pandas as pd
import numpy as np
import time

class BacktestEngine:
    """
    Backtesting engine for Hyperliquid quant strategies.
    Evaluates the SOTA TimeSeriesFoundationModel on historical data.
    """
    def __init__(self, model: nn.Module, coin: str = "SOL"):
        self.model = model
        self.coin = coin
        self.ingestor = HyperliquidDataIngestor(coin=coin)
        self.model.eval()

    @torch.no_grad()
    def run_backtest(self, seq_len: int = 100):
        """
        Runs a simple backtest by predicting next-step returns.
        """
        print(f"Starting backtest for {self.coin}...")
        
        # 1. Get data
        inputs, targets = self.ingestor.prepare_training_shard(seq_len=seq_len, batch_size=128)
        
        # 2. Predict
        predictions = self.model(inputs)
        
        # 3. Calculate metrics
        mse = torch.nn.functional.mse_loss(predictions, targets).item()
        
        # Simple directional accuracy
        pred_dir = torch.sign(predictions)
        target_dir = torch.sign(targets)
        accuracy = (pred_dir == target_dir).float().mean().item()
        
        # Simulate cumulative returns (very basic)
        # If prediction > 0, go long; if prediction < 0, go short
        strategy_returns = pred_dir * targets
        cum_returns = torch.cumsum(strategy_returns, dim=0)
        
        results = {
            "mse": mse,
            "directional_accuracy": accuracy,
            "total_return": cum_returns[-1].item() if cum_returns.numel() > 0 else 0,
            "sharpe_ratio": (strategy_returns.mean() / (strategy_returns.std() + 1e-8)).item() * np.sqrt(252) # Annualized (rough)
        }
        
        return results

if __name__ == "__main__":
    from quant.quant_model import get_quant_model
    model = get_quant_model()
    engine = BacktestEngine(model)
    results = engine.run_backtest()
    print("Backtest Results:", results)
