# Hybrid Training Summary

Training Date: 2025-11-10 09:23:10.428099

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
- Iterations: 400
- Episodes per iter: 20
- Final avg reward: 5.88

## Behavior Conditioning Results
- S-L: ax=0.462, ay=3.243
- S-R: ax=0.735, ay=0.618
- L-R: ax=1.042, ay=3.483
