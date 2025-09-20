import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.multiprocessing as mp

def train(rank, world_size):
    # Set the GPU device for THIS rank
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Initialize DDP process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Initialized on device {device}", flush=True)

    # Simple model
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[rank])

    # Toy dataset
    dataset = TensorDataset(torch.randn(32, 10), torch.randn(32, 1))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(15):
        sampler.set_epoch(epoch)  # shuffle differently each epoch
        print(f"[Epoch {epoch}] Rank {rank} starting training ({len(dataloader)} batches)", flush=True)

        epoch_loss_accum = torch.tensor(0.0, device=device)
        for batch_idx, (x, y) in enumerate(dataloader):
            start_time = time.time()

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            # Reduce loss across GPUs
            loss_all = loss.detach().clone()
            dist.all_reduce(loss_all, op=dist.ReduceOp.SUM)
            loss_all /= world_size
            epoch_loss_accum += loss_all

            batch_time = time.time() - start_time
            print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss={loss.item():.4f}, Step Time={batch_time:.3f}s", flush=True)

        epoch_loss_avg = epoch_loss_accum / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch}] >>> Average Loss: {epoch_loss_avg.item():.4f}", flush=True)

    # Cleanup
    dist.destroy_process_group()
    print(f"[Rank {rank}] Finished training", flush=True)


if __name__ == "__main__":
    # Set environment variables for single-node DDP
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    if torch.cuda.is_available():
        print("GPUS:", torch.cuda.device_count())
    else:
        print("Cuda is not available.")
        exit(1)

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
