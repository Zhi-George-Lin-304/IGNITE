# IGNITE
<img width="204" height="204" alt="image" src="https://github.com/user-attachments/assets/218055a3-5084-4b1e-9f33-d00ab8259525" />


🧬 Molecular Reinforcement Learning Generator
LSTM + GCN/MLP Property Predictor + Reinforcement Learning

This repository implements a molecular generative model trained with reinforcement learning, where:

● The LSTM autoregressively generates SMILES strings

● A GCN + MLP model predicts a target molecular property (e.g., singlet–triplet gap)

● A REINFORCE RL loop updates the LSTM to generate molecules with desirable properties

The full workflow is:

SMILES → LSTM → generate candidates → GCN+MLP → property prediction → reward → update LSTM

This README explains how to install, train, generate, and extend the system.
