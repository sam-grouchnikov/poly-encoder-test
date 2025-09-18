import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# Crucial setup for allowing cross-gpu communication
def setup():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

    print(f"[SETUP] rank={rank}, local_rank={local_rank}, world_size={world_size}, device={torch.cuda.current_device()}")
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()


def train():
    rank, local_rank, world_size = setup()

    model = nn.Linear(10, 1).to(local_rank)
    # Wrapping the model as a DDP
    model = DDP(model, device_ids=[local_rank])

    dataset = TensorDataset(torch.randn(32, 10), torch.randn(32, 1))
    # Distributed the data evenly across GPUs
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(2):
        sampler.set_epoch(epoch)  # shuffle differently each epoch
        print(f"\n[Epoch {epoch}] Rank {rank} starting training loop with {len(dataloader)} batches")
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(local_rank), y.to(local_rank)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss={loss.item():.4f}")

        if rank == 0:
            print(f"[Epoch {epoch}] >>> Final loss (Rank 0 only): {loss.item():.4f}")

    # Shuts down process group
    cleanup()

train()

# torchrun --standalone --nproc_per_node=(number of gpus) PytorchParallelComputing.py

# Expected with 2 gpus
# [Epoch x] Rank 0 starting training loop with 4 batches
# [Epoch x] Rank 1 starting training loop with 4 batches

# Expected with 4 gpus
# [Epoch x] Rank 0 starting training loop with 2 batches
# [Epoch x] Rank 1 starting training loop with 2 batches
# [Epoch x] Rank 2 starting training loop with 2 batches
# [Epoch x] Rank 3 starting training loop with 2 batches