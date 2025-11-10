# Hybrid Training Summary

Training Date: 2025-11-09 16:56:12.240058

## Configuration
- Data samples: 450,000
- State dim: 34
- Action dim: 2
- Policy params: 34,212

## Stage 1: BC
- Epochs: 100
- Batch size: 256
- Best val loss: 1.6398

## Stage 2: RL
- Iterations: 200
- Episodes per iter: 20
- Final avg reward: 0.21

## Behavior Conditioning Results
- S-L: ax=0.323, ay=2.742
- S-R: ax=0.363, ay=1.262
- L-R: ax=0.245, ay=1.760
