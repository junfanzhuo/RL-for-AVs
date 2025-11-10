# Hybrid Training Summary

Training Date: 2025-11-10 09:16:57.239201

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
- Final avg reward: 0.41

## Behavior Conditioning Results
- S-L: ax=0.562, ay=3.226
- S-R: ax=0.469, ay=1.158
- L-R: ax=0.157, ay=2.410
