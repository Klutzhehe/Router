"""
PCB Router — AI-powered PCB routing with GNN + JEPA architecture.

Uses a Graph Neural Network for netlist topology encoding, a JEPA world-model
for spatial reasoning, and heatmap-guided A* for DRC-compliant trace generation.
Trained via curriculum-based reinforcement learning (PPO).
"""

__version__ = "0.1.0"
