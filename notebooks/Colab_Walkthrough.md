# Google Colab Training Walkthrough

This guide explains how to use the **`notebooks/Train_PCB_Router.ipynb`** Jupyter notebook in Google Colab to run Behavior Cloning pretraining and Dreamer RL fine-tuning.

---

## Step 1: Open the Notebook in Google Colab

1. Upload the `notebooks/Train_PCB_Router.ipynb` file to your Google Drive or commit it to your GitHub repository.
2. Go to [Google Colab](https://colab.research.google.com/).
3. Choose **File > Open notebook** and select the notebook from Google Drive or load it directly from your GitHub URL.
4. **Enable GPU acceleration**:
   - Go to **Runtime > Change runtime type**.
   - Under *Hardware accelerator*, select **T4 GPU** (or A100/L4 if available).
   - Click **Save**.

---

## Step 2: Cell 2 — Edit your Configurations (`CONFIG`)

Navigate to **CELL 2** and configure the variables:

- **`REPO_URL`**: Set this to your GitHub repository URL:
  ```python
  "REPO_URL": "https://github.com/your-username/your-repo-name.git"
  ```
- **`CHECKPOINT_DIR`**: Recommended to keep the default Drive path so checkpoints survive Colab runtime timeout disconnects:
  ```python
  "CHECKPOINT_DIR": "/content/drive/MyDrive/pcb_router/checkpoints"
  ```
- **`USE_WANDB`**: (Optional) Set to `True` and add your project name if you want to track training curves live via Weights & Biases.

---

## Step 3: Run the Cells Sequentially

### 1. Cell 3: Mount Google Drive
- Run this cell to authorize Google Colab to read/write checkpoints to your Google Drive folder.

### 2. Cell 4: Git Clone & Dependency Installation
- Run this cell. It will clone your repo into `/content/Router`, set up PyTorch Geometric (PyG), PyTorch Scatter, Gymnasium, and other dependencies. This take about 2-3 minutes.

### 3. Cell 4b: Expert Dataset Generation & BC Pretraining
- **Run this cell to start pretraining**:
  - `generate_bc_dataset.py` will generate expert routes across stages `s00-s06`.
  - `train_bc_policy.py` will train the `RouteStepPolicy` using supervised cross-entropy with dynamic move validation masking.
  - The pretrained weights are automatically saved to `checkpoints/bc_pretrained_policy.pt`.

### 4. Cell 5: Initialize the Trainer
- Run this cell to load the curriculum configurations, compile the model architectures, and initialize the `DreamerJEPATrainer`.
- The trainer automatically checks for the pre-trained BC policy weights in the checkpoint folder and loads them as the default step actor.

### 5. Cell 6: Train with Live Visuals
- Run this cell to start the Dreamer world model and RL imagination loop.
- It displays a live training dashboard containing:
  - Routing Completion Rate.
  - Actor Loss.
  - JEPA World Model Loss.
  - Critic Loss.
  - A real-time rendering of the current board's routed layout.
