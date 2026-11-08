# Full 100-epoch SAGE run on the DGX (JupyterLab)

The DGX is a Linux server with NVIDIA GPUs. It has none of Colab's limits: no daily GPU allowance, and no disconnects when you close the browser. Everything below happens in a **Terminal** inside JupyterLab.

## 1. Open a terminal and check the GPUs (1 minute)

In JupyterLab, click **File → New → Terminal**, then type:

```bash
nvidia-smi
```

This lists the GPUs (numbered 0, 1, 2, ...), with how much memory each is using. Other people may be using the server too, so pick a GPU whose **Memory-Usage** is low, for example GPU 3, and use that number in step 4.

## 2. Get the code (2 minutes)

```bash
cd ~
git clone https://github.com/Floxy1991/Sage.git
cd Sage
```

If `git` is missing or the server can't reach GitHub, download the ZIP on your own computer (GitHub → Code → Download ZIP). Then upload it with JupyterLab's upload button (the ↑ arrow in the file panel) and run `unzip Sage-main.zip && mv Sage-main Sage && cd Sage`.

## 3. Install (5–10 minutes)

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- If this prints `True`, PyTorch with GPU support is already installed. Run:
  ```bash
  pip install --user transformers sentencepiece protobuf accelerate nltk scikit-learn rank-bm25 psutil pandas
  ```
- If it prints `False` or `No module named torch`, run:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install torch transformers sentencepiece protobuf accelerate nltk scikit-learn rank-bm25 psutil pandas
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  This must print `True`. On Linux, the normal `pip install torch` already includes GPU support.

If `pip` can't download anything, the server has no internet access. Ask your supervisor how packages and Hugging Face models are installed on it.

## 4. Start the run (1 minute)

Replace `3` with the GPU number you chose in step 1:

```bash
CUDA_VISIBLE_DEVICES=3 nohup bash run_paper_gpu.sh > train_log.txt 2>&1 &
```

`nohup ... &` keeps it running after you close the browser or the virtual desktop. To watch progress:

```bash
tail -f train_log.txt
```

You'll see lines like `epoch 1/100: batch 400/2325, 3 min, loss 7.1`. Press **Ctrl+C** to stop *watching*. This doesn't stop the training.

## 5. Coming back later

```bash
cd ~/Sage
tail -n 20 train_log.txt
```

- **`ALL DONE`** at the end: the results are in `results/sage_paper100/metrics.json`, and the comparison table with the paper is just above. Send a screenshot.
- **Still printing epochs:** it's still running. Leave it.
- **An error, or the server restarted:** run the step 4 command again. It continues from the last finished epoch. If the same error comes back, send a screenshot.

To check it's still running: `ps aux | grep sage.py`. To stop it on purpose: `pkill -f sage.py`.

## Out of memory

If `train_log.txt` ends with `CUDA out of memory`, the GPU you picked was too busy. Choose a freer GPU in `nvidia-smi` and run step 4 again with that number.
