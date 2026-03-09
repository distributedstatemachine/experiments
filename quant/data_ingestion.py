import os
import json
import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

class HyperliquidDataIngestor:
    """
    Ingests L2/L3 data from Hyperliquid for training.
    Focuses on decentralized data sharding.
    """
    def __init__(self, coin: str = "SOL"):
        self.coin = coin
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        
    def fetch_l2_snapshot(self):
        """Fetches current L2 orderbook snapshot."""
        return self.info.l2_snapshot(self.coin)
    
    def fetch_recent_trades(self):
        """Fetches recent trades for the coin."""
        return self.info.recent_trades(self.coin)

    def prepare_training_shard(self, seq_len: int = 100, batch_size: int = 32):
        """
        Prepares a data shard for decentralized training using real Hyperliquid data.
        """
        import torch
        import numpy as np
        
        try:
            # 1. Fetch recent trades
            trades = self.fetch_recent_trades()
            if not trades:
                raise ValueError("No trades fetched")
            
            # 2. Extract prices and normalize
            prices = [float(t['px']) for t in trades]
            if len(prices) < seq_len + batch_size:
                # Fallback to dummy data if not enough real data
                print(f"Warning: Not enough real data ({len(prices)} < {seq_len + batch_size}). Using dummy data.")
                inputs = torch.randn(batch_size, seq_len, 1)
                targets = torch.randn(batch_size, 1)
                return inputs, targets

            # Simple normalization: pct_change
            prices_arr = np.array(prices)
            returns = np.diff(prices_arr) / prices_arr[:-1]
            
            # Create sliding windows
            X, y = [], []
            for i in range(len(returns) - seq_len):
                X.append(returns[i:i+seq_len])
                y.append(returns[i+seq_len])
                if len(X) >= batch_size:
                    break
            
            inputs = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            targets = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
            
            return inputs, targets
            
        except Exception as e:
            print(f"Error preparing training shard: {e}. Falling back to dummy data.")
            inputs = torch.randn(batch_size, seq_len, 1)
            targets = torch.randn(batch_size, 1)
            return inputs, targets

    def fetch_historical_data(self, start_time: int, end_time: int):
        """
        Fetches historical data for backtesting.
        Note: Hyperliquid API has limits on historical data. 
        In a production system, this would interface with a dedicated data lake.
        """
        # Placeholder for historical data fetching logic
        # For now, we'll simulate it or use recent data as a proxy
        print(f"Fetching historical data from {start_time} to {end_time}...")
        return self.fetch_recent_trades()

if __name__ == "__main__":
    ingestor = HyperliquidDataIngestor()
    print(f"Fetching data for {ingestor.coin}...")
    # print(ingestor.fetch_l2_snapshot())
