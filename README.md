Federated Deep Generative Models

This repository explores how federated learning and deep generative models (GANs) can work together.
Instead of training GANs on a single centralized dataset, multiple clients train locally and share their updates with a central server. This approach protects data privacy while still producing powerful generative models.

The project includes three main notebooks: 1. Federated Conditional DCGAN – a conditional DCGAN trained across federated clients. 2. FedGAN v2 (Kaggle Run) – a clean, reproducible version designed to run easily on Kaggle GPUs. 3. FedGAN with Custom Synchronization – experiments with alternative aggregation rules beyond simple averaging.

⸻

Why This Project Matters

Federated learning is widely used for classification tasks, but applying it to generative modeling is less common. These experiments demonstrate how GANs can be adapted to federated settings, enabling applications such as privacy-preserving image generation in fields like healthcare, finance, and other sensitive domains.

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

⸻

Getting Started

Local Setup

conda create -n fedgan python=3.10 -y
conda activate fedgan
pip install torch torchvision torchaudio matplotlib tqdm scikit-learn torchmetrics

Install the CUDA-enabled version of PyTorch if you have a GPU; otherwise, use the CPU build.

Kaggle Setup

Open the FedGAN v2 Kaggle notebook, enable GPU, and run all cells. No additional setup is required.

⸻

Datasets

The default dataset is MNIST (handwritten digits).
It will be downloaded automatically. You can switch to Fashion-MNIST or CIFAR-10 by adjusting the dataset code in the notebooks.

⸻

What to Expect
• Conditional samples generated per digit (0–9).
• Training curves showing generator and discriminator progress across rounds.
• Comparisons between standard federated averaging and custom synchronization strategies.

⸻

Tips for Training
• If samples are blurry or repetitive → reduce discriminator learning rate.
• If training is unstable → lower the number of local epochs or add small amounts of noise.
• For more realistic outputs → increase the number of federated rounds.

⸻

Possible Extensions
• Test on non-IID client partitions.
• Add privacy-preserving mechanisms (e.g., differential privacy).
• Try different aggregation methods (FedProx, FedAdam, etc.).
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
