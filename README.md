# DDP_Demo
Pytorch DDP demo

training scripts

For single gpu

```bash
python train_single_gpu.py --device {gpu_id} --batch_size 32
```

For multi-gpu 

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_ddp_single_node.py  --batch_size 32
```
