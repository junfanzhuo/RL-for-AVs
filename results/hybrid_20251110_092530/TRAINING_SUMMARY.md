# Hybrid Training Summary

Training Date: 2025-11-10 09:26:07.781711

## Configuration
- Data samples: 450,000
- State dim: 34
- Action dim: 2
- Policy params: 34,212

## Stage 1: BC
- Epochs: 100
- Batch size: 256
- Best val loss: 1.8371

## Stage 2: RL
- Iterations: 200
- Episodes per iter: 20
- Final avg reward: 2.42

## Behavior Conditioning Results
- S-L: ax=0.337, ay=3.587
- S-R: ax=0.039, ay=4.259
- L-R: ax=0.298, ay=7.846
