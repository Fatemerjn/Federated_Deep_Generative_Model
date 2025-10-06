# Deep Federated Generative Models — (cDCGAN, FedGAN v2, Custom Sync)

This repository hosts three complementary notebooks demonstrating **federated learning for generative models** using PyTorch:

1. **`Federated_conditional_DCGAN.ipynb`** — Conditional DCGAN trained in a federated setting (clients + FedAvg), class-conditional sampling.
2. **`Deep_Federated_Generative_Models__FedGAN_v2_kaggle_run.ipynb`** — A Kaggle‑friendly FedGAN v2 pipeline with clean logging and reproducible outputs.
3. **`FedGAN_Custom_sync_strategy.ipynb`** — A variant exploring custom synchronization/aggregation strategies on top of standard FedAvg.

> If you are new to FL + GANs: start with **`Federated_conditional_DCGAN.ipynb`**, then run the **Kaggle v2** notebook, and finally experiment with **Custom Sync**.

---

## ✨ Highlights

- **Federated Simulation**: Multiple clients train locally; server aggregates (FedAvg by default).
- **Conditional Generation**: Class‑conditional DCGAN capable of label‑controlled synthesis.
- **Kaggle‑Ready**: A notebook that runs out‑of‑the‑box on Kaggle GPU with minimal path assumptions.
- **Extensible**: Pluggable aggregation (FedAvg → FedProx/FedAdam/custom), non‑IID splits, DP.
- **Clear Outputs**: Loss curves, image grids, and optional metrics (IS/FID) hooks.

---

## 📦 Repository Structure

```
.
├─ Federated_conditional_DCGAN.ipynb
├─ Deep_Federated_Generative_Models__FedGAN_v2_kaggle_run.ipynb
├─ FedGAN_Custom_sync_strategy.ipynb
├─ README.md  ← you are here
└─ (created at runtime)
   ├─ outputs/           # generated samples, logs, model weights
   └─ checkpoints/       # optional torch.save(...) artifacts
```

You can keep everything notebook‑centric; the directories above are created by the notebooks if missing.

---

## 🧰 Environment & Setup

### Option A — Conda (recommended)
```bash
conda create -n fedgan python=3.10 -y
conda activate fedgan
# Choose a torch build suitable to your machine (CPU/GPU).
# CUDA 11.8 example (Linux/Windows with NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU-only example:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install numpy matplotlib tqdm scikit-learn torchmetrics
```

### Option B — Kaggle
- Open **`Deep_Federated_Generative_Models__FedGAN_v2_kaggle_run.ipynb`** in a Kaggle Notebook.
- Turn on **GPU** (if available) and run all cells in order. No extra path setup should be required.

> Python 3.9+ is recommended. CUDA support is optional but speeds up training significantly.

---

## 📚 Datasets

The notebooks default to **MNIST** via `torchvision` and will auto‑download:
- 10 classes (digits 0–9), 28×28 grayscale
- Used for both **federated partitioning** and **conditional** generation in cDCGAN

You can easily switch to **Fashion‑MNIST** or other torchvision datasets by editing the data section.
For **non‑IID** experiments, label‑skewed splits per client can be enabled (see the Custom Sync notebook for hooks).

---

## 🚀 How to Run

### 1) Federated Conditional DCGAN
Notebook: **`Federated_conditional_DCGAN.ipynb`**

- Implements a conditional DCGAN (`G`/`D` with label conditioning) trained across **K** clients.
- **Federated rounds**:
  1. Server broadcasts global params
  2. Clients train locally for `E` epochs
  3. Server aggregates (FedAvg) → new global
  4. Log losses and sample conditioned grids

**Key Config (top cells):**
- `num_clients`, `rounds`, `local_epochs`
- `batch_size`, `lrG`, `lrD`, `betas`
- `z_dim`, conditional embedding size, image size
- `device` (`cuda`/`cpu`)

**Outputs:**
- Grids per label (0–9), loss curves
- Optional model checkpoints under `checkpoints/`

---

### 2) FedGAN v2 — Kaggle Run
Notebook: **`Deep_Federated_Generative_Models__FedGAN_v2_kaggle_run.ipynb`**

- A clean Kaggle pipeline with deterministic seeds and simple logging.
- Same fed cycle (broadcast → local train → aggregate), plus quality‑of‑life utilities for plotting and saving.

**Tips:**
- Ensure Kaggle **GPU** is enabled for faster runs.
- Adjust `K`, `rounds`, and `local_epochs` for runtime budget.
- Metrics hooks for IS/FID are provided (enable as needed; may require `torchmetrics` + `scipy`).

**Outputs:**
- Inline images per round, loss curves, and optional saved weights.

---

### 3) Custom Synchronization Strategy
Notebook: **`FedGAN_Custom_sync_strategy.ipynb`**

- Demonstrates **alternative server update rules** (e.g., weighted/thresholded updates, damped moving averages, or client selection policies).
- Useful for **straggler mitigation**, **communication constraints**, or **stability** experiments in GAN training.

**What to tweak:**
- The **aggregator** function (e.g., switch from pure FedAvg to a custom rule)
- **Client sampling** per round (uniform, proportional to data volume, or skewed)
- **Non‑IID** splits

**Outputs:**
- Loss curves and samples (add more logging if you run ablations)

---

## 📏 Metrics (optional but recommended)

GANs benefit from **qualitative** and **quantitative** checks:

- **Qualitative**: Fixed‑noise sample grids, per‑label grids (cDCGAN); visual sharpness/variety over rounds.
- **Quantitative**:
  - **IS** (Inception Score): higher is generally better (watch for mode collapse).
  - **FID** (Fréchet Inception Distance): lower is better (requires feature extractor; more setup).

> The notebooks include hooks/snippets for these metrics. Enable as needed. For strict comparability, fix seeds and dataset partitions.

---

## ⚙️ Configuration Cheatsheet

Common hyperparameters (set near the top of each notebook):
- `num_clients`: 5–20 for MNIST demos
- `rounds`: 50–200 (increase for higher fidelity)
- `local_epochs`: 1–5 (more local steps = fewer comms but may drift)
- `batch_size`: 64–256
- `z_dim`: 64–128 (latent code)
- `lrG`, `lrD`: typically `2e-4` with Adam `betas=(0.5, 0.999)` for DCGAN‑style training

Federated specifics:
- **Aggregator**: FedAvg (baseline), custom rule in the Custom Sync notebook
- **Client sampling**: all or a subset each round
- **Weighting**: by client dataset size (recommended)

---

## 🧪 Reproducibility

- Set all seeds (`random`, `numpy`, `torch`, `cuda`).
- Log the exact **Torch**, **CUDA**, and **driver** versions.
- Save configs and checkpoints with round indices.
- For Kaggle: record the notebook Docker image hash (visible in the UI).

---

## 🛠️ Troubleshooting

- **Blurry or identical samples**: reduce `lrD`, slightly increase `lrG`, or increase training rounds; verify conditioning pipeline.
- **Discriminator overpowering**: lower `lrD`, use label smoothing, or add instance noise.
- **Destabilization with many local epochs**: decrease `local_epochs` or add server‑side damping in the aggregator.
- **Inconsistent client sizes**: use size‑weighted aggregation (FedAvg default) and consider stratified client sampling.

---

## 🔌 Extending the Notebooks

- Swap MNIST with **Fashion‑MNIST** or **CIFAR‑10** (update model size/arch for RGB).
- Add **Differential Privacy** (e.g., Opacus) for private FL.
- Try **FedProx**, **FedAdam**, or **scaffold‑like** corrections.
- Explore **non‑IID** partitions: dirichlet label skew, quantity skew, real client logs.
- Add **projected discriminator** or **spectral norm** for stronger cGAN baselines.

---

## 📖 References

- McMahan et al., *Communication‑Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 (FedAvg)  
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014  
- Radford et al., *Unsupervised Representation Learning with DCGAN*, ICLR 2016  
- Mirza & Osindero, *Conditional Generative Adversarial Nets*, 2014  
- Heusel et al., *GANs Trained by a Two Time‑Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017 (FID)

---

## 📜 License

Unless otherwise stated, this project is released under the **MIT License**. Update the license file if you use another license.