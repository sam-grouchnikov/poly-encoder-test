import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def main():
    # Get rank and world size from torchrun environment
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    print(f"[Rank {rank}] Using device {device}", flush=True)

    # Initialize DDP
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Simple model
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Toy dataset
    dataset = TensorDataset(torch.randn(32, 10), torch.randn(32, 1))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(15):
        sampler.set_epoch(epoch)
        epoch_loss_accum = torch.tensor(0.0, device=device)

        print(f"[Epoch {epoch}] Rank {rank} starting ({len(dataloader)} batches)", flush=True)
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

        # Print epoch loss on rank 0
        epoch_loss_avg = epoch_loss_accum / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch}] >>> Average Loss: {epoch_loss_avg.item():.4f}", flush=True)

    dist.destroy_process_group()
    print(f"[Rank {rank}] Finished training", flush=True)


if __name__ == "__main__":
    main()
