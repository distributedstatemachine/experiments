import torch
import torch.nn as nn
import torch.optim as optim
from quentin.sparseloco import SparseLoCoOptimizer, BasilicaAggregator, SPoTVerifier
import time
import random

# Simple model for simulation
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

def run_simulation(num_workers=4, steps_per_worker=10, density=0.1):
    print(f"Starting Basilica SparseLoCo Simulation (Workers: {num_workers}, Density: {density})")
    
    global_model = SimpleNet()
    aggregator = BasilicaAggregator(global_model)
    verifier = SPoTVerifier(SimpleNet, density=density)
    
    workers = []
    for i in range(num_workers):
        worker_model = SimpleNet()
        worker_model.load_state_dict(global_model.state_dict())
        optimizer = SparseLoCoOptimizer(worker_model.parameters(), density=density)
        workers.append({
            'id': f"worker_{i}",
            'model': worker_model,
            'optimizer': optimizer,
            'local_opt': optim.SGD(worker_model.parameters(), lr=1e-3) # Use SGD for deterministic replay
        })

    # Simulate training rounds
    for round_idx in range(5):
        print(f"\n--- Round {round_idx} ---")
        
        # Track average loss for convergence monitoring
        total_loss = 0
        
        for worker in workers:
            # 1. Local training (H steps)
            seed = round_idx * 1000 + int(worker['id'].split('_')[1])
            torch.manual_seed(seed)
            data = torch.randn(10, 10)
            target = torch.randn(10, 10)
            
            initial_weights = [p.data.clone().detach() for p in worker['model'].parameters()]
            
            worker_loss = 0
            for _ in range(steps_per_worker):
                worker['local_opt'].zero_grad()
                output = worker['model'](data)
                loss = nn.MSELoss()(output, target)
                loss.backward()
                worker['local_opt'].step()
                worker_loss += loss.item()
            
            total_loss += worker_loss / steps_per_worker
            
            # 2. Compute sparse update (now with 2-bit quantization)
            bits, idxs, scales = worker['optimizer'].get_sparse_update()
            
            # 3. Simulate latency and churn
            if random.random() < 0.1:
                print(f"Worker {worker['id']} dropped out this round.")
                continue
                
            # 4. Verification (SPoT)
            is_valid = verifier.verify_update(
                initial_weights, bits, idxs, scales, (data, target), 
                steps_per_worker, 1e-3, seed
            )
            
            if not is_valid:
                print(f"Worker {worker['id']} failed verification! Slashing...")
                continue
            else:
                print(f"Worker {worker['id']} passed SPoT verification.")

            # 5. Push to aggregator (The Citadel)
            # Simulate staleness: worker might be behind
            worker_version = aggregator.global_version - random.randint(0, 2)
            aggregator.apply_sparse_update(bits, idxs, scales, worker['id'], worker_version)
            
            # 6. Synchronize (Pull from Citadel)
            worker['optimizer'].synchronize(aggregator.get_global_weights())
            
            # Reset local optimizer state for the new weights
            worker['local_opt'] = optim.SGD(worker['model'].parameters(), lr=1e-3)
            
        print(f"Global model updated. Version: {aggregator.global_version}, Avg Loss: {total_loss/num_workers:.4f}")

if __name__ == "__main__":
    run_simulation()
