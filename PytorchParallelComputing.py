import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.multiprocessing as mp
import wandb


def train(rank, world_size):
    # Setup environment for DDP
    print(f"Rank {rank}: CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Rank {rank}: Current device: {torch.cuda.current_device()}", flush=True)
    print(f"Rank {rank}: Device count: {torch.cuda.device_count()}", flush=True)
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{}'.format(rank))
    print("Intializing")
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # local host for single-node
    os.environ['MASTER_PORT'] = '12355'      # any free port
    print("Initialized")
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    print(f"[SETUP] rank={rank}, device={torch.cuda.current_device()}, world_size={world_size}")

    if rank == 0:
        wandb.init(project="ddp_debug", name="toy_ddp_run")
        wandb.config.update({"world_size": world_size, "batch_size": 2, "dataset_size": 32})

    # Model as DDP process
    model = nn.Linear(10, 1).to(rank)
    model = DDP(model, device_ids=[rank])


    dataset = TensorDataset(torch.randn(32, 10), torch.randn(32, 1))
    # Allows data to be evenly distributed across GPUs
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # Batch size is per step, so this may be changed without affecting distributed computing
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(15):
        sampler.set_epoch(epoch)  # ensure shuffling is different per epoch
        print(f"\n[Epoch {epoch}] Rank {rank} starting training with {len(dataloader)} batches")

        epoch_loss_accum = torch.tensor(0.0).to(rank)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss={loss.item():.4f}")

            loss_all = loss.clone()
            dist.all_reduce(loss_all, op=dist.ReduceOp.SUM)
            loss_all /= world_size  # mean loss across all GPUs
            epoch_loss_accum += loss_all  # accumulate for epoch

            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss={loss.item():.4f}")

            if rank == 0:
                wandb.log({
                    "loss_step": loss_all.item(),
                    "step": epoch * len(dataloader) + batch_idx
                })
        epoch_loss_avg = epoch_loss_accum / len(dataloader)  # average across batches
        if rank == 0:
            wandb.log({
                "loss_epoch": epoch_loss_avg.item(),
                "epoch": epoch
            })
            print(f"[Epoch {epoch}] >>> Rank 0 epoch loss: {epoch_loss_avg.item():.4f}")

    if rank == 0:
        wandb.finish()
    # Cleanup
    dist.destroy_process_group()
    print(f"Rank {rank} finished training.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPUS: ", torch.cuda.device_count())
    else:
        print("Cuda is not available.")
    world_size = torch.cuda.device_count() # number of GPUs / processes
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

# torchrun --standalone --nproc_per_node=4 PytorchParallelComputing.py

# Expected with 2 gpus
# [Epoch x] Rank 0 starting training loop with 4 batches
# [Epoch x] Rank 1 starting training loop with 4 batches

# Expected with 4 gpus
# [Epoch x] Rank 0 starting training loop with 2 batches
# [Epoch x] Rank 1 starting training loop with 2 batches
# [Epoch x] Rank 2 starting training loop with 2 batches
# [Epoch x] Rank 3 starting training loop with 2 batches