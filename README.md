Federated Deep Generative Models

This repository explores how federated learning and deep generative models (GANs) can work together.
Instead of training GANs on one central dataset, multiple clients train locally and share their updates with a central server. This setup protects data privacy while still producing powerful generative models.

The project includes three main notebooks: 1. Federated Conditional DCGAN – a conditional DCGAN trained across federated clients. 2. FedGAN v2 (Kaggle Run) – a clean, reproducible version designed to run easily on Kaggle GPUs. 3. FedGAN with Custom Synchronization – experiments with alternative aggregation rules beyond simple averaging.

⸻

Why This Project Matters

Federated learning is commonly used for classification and prediction tasks, but training generative models in this setting is less explored. These experiments demonstrate how GANs can be adapted to federated settings, opening opportunities for applications such as privacy-preserving image generation in healthcare, finance, and other sensitive domains.

⸻

Repository Structure

.
├─ notebooks/
│ ├─ Federated_conditional_DCGAN.ipynb
│ ├─ Deep_Federated_Generative_Models\_\_FedGAN_v2_kaggle_run.ipynb
│ └─ FedGAN_Custom_sync_strategy.ipynb
├─ outputs/ # generated images & logs
├─ checkpoints/ # saved models
└─ README.md

All the main experiments are in the notebooks. Supporting directories are created when you run them.

⸻

Getting Started

Local Setup

conda create -n fedgan python=3.10 -y
conda activate fedgan
pip install torch torchvision torchaudio matplotlib tqdm scikit-learn torchmetrics

Install the CUDA-enabled version of PyTorch if you have a GPU; otherwise, use the CPU version.

Kaggle Setup

Open the FedGAN v2 Kaggle notebook, enable GPU, and run all cells. No additional setup is needed.

⸻

Datasets

Default experiments use MNIST (handwritten digits).
It will be downloaded automatically. You can switch to Fashion-MNIST or CIFAR-10 by editing the dataset section of the notebooks.

⸻

What to Expect
• Conditional samples generated per digit (0–9).
• Training curves showing generator and discriminator progress across rounds.
• Comparisons between standard federated averaging and custom synchronization strategies.

⸻

Tips for Training
• Blurry or repeated samples → reduce discriminator learning rate.
• Instability during training → lower the number of local epochs or add small amounts of noise.
• More realistic outputs → increase the number of federated rounds.

⸻

Extensions
• Experiment with non-IID client partitions.
• Add privacy mechanisms such as differential privacy.
• Test additional aggregation algorithms (FedProx, FedAdam, etc.).
• Replace MNIST with more complex datasets.

⸻

References
• McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg, 2017)
• Goodfellow et al., Generative Adversarial Nets (GANs, 2014)
• Radford et al., Unsupervised Representation Learning with DCGAN (2016)
• Mirza & Osindero, Conditional Generative Adversarial Nets (2014)

⸻

License

This project is released under the MIT License.

⸻

do you want me to also prepare a shorter summary version (about one-third of this length) that you could use as the GitHub repo front-page description?
