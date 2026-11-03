# Full 100-epoch SAGE run on a Windows PC with an NVIDIA GPU

This runs SAGE with the paper's own settings (100 self-retrieval epochs) directly on the computer, without Colab, so it can run for days with nobody watching. Everything happens in one file, `run_paper_gpu.bat`.

## 1. Check the GPU (2 minutes)

1. Open **Command Prompt**: press the Windows key, type `cmd`, press Enter.
2. Type `nvidia-smi` and press Enter.
   - A table with a card name (for example "RTX 4090") and its memory (for example "24564MiB") means it can be used. Note the **CUDA Version** in the top right.
   - "'nvidia-smi' is not recognized" means there's no usable NVIDIA card. Stop here and use Colab.
3. You need **at least 12 GB** of GPU memory. With 8 GB, see "Out of memory" below.

How long it takes depends on the card. On Colab's T4, one epoch takes about 55 minutes. A newer card such as an RTX 3090, 4090 or A100 should be several times faster, so 100 epochs could take roughly 1 to 2 days. This is an estimate: the first progress lines will show the real speed.

## 2. One-time set-up (about 20 minutes)

1. **Python:** in Command Prompt, type `python --version`. If it prints 3.10, 3.11 or 3.12, skip ahead. Otherwise install Python 3.12 from python.org. Tick **"Add python.exe to PATH"**. If you have no admin rights, choose "Install for me only" (not "for all users").
2. **The code:** go to https://github.com/Floxy1991/Sage, click the green **Code** button, then **Download ZIP**. Unzip it somewhere with plenty of free space (10 GB), for example `C:\Users\<you>\Sage`. If the school PC has Git, `git clone https://github.com/Floxy1991/Sage.git` also works.
3. **Install the packages.** In Command Prompt:
   ```
   cd C:\Users\<you>\Sage
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements-local.txt
   ```
   The first `pip install torch` line installs the **GPU** version of PyTorch, and it must come before the second line. If `nvidia-smi` showed a CUDA Version below 12.4, use `cu121` or `cu118` instead of `cu124`.
4. **Check it sees the GPU:**
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   It must print `True` and the card's name. If it prints `False`, uninstall it with `pip uninstall -y torch` and repeat the `cu124` line.

## 3. Start the run

Double-click **`run_paper_gpu.bat`** in the Sage folder, or type `run_paper_gpu.bat` in Command Prompt in that folder.

It prints the GPU name, then progress lines like `epoch 1/100: batch 400/2325, 6 min, loss 7.1`. The loss stays near 7 for the first few epochs while the learning rate warms up, and then it should fall.

**Before you leave it running:**
- Settings → System → Power: set sleep to **Never**, and keep it plugged in.
- **Don't log off or shut down.** You can lock the screen (Windows + L). Locking is fine, but logging off stops the run.
- Ask IT whether the PC restarts at night or for updates. If it does, the run stops but nothing is lost (see below).
- Leave the Command Prompt window open.

## 4. When you come back

- **The window says "ALL DONE":** the results are in `results\sage_paper100\metrics.json`, and a comparison table with the paper is printed above. Send a screenshot or the file.
- **The window shows an error, or the PC restarted:** double-click `run_paper_gpu.bat` again. It continues from the last finished epoch, so at most one epoch is lost. If the same error comes back, send a screenshot.
- **It's still training:** check the latest `epoch N/100` line, and leave it running.

To move the trained models elsewhere (for example to evaluate on your laptop), copy the `ckpt` folder. It's about 2.7 GB.

## Out of memory

If the error says `CUDA out of memory`:
1. Open `run_paper_gpu.bat` in Notepad.
2. In the train-selfret line, change `--bs 16 --grad-accum 8` to `--bs 8 --grad-accum 16`. The effective batch stays 128, so the result is the same, but it's a bit slower.
3. If it fails during fusion training, change `--micro-bs 4` to `--micro-bs 2`.
4. Save, and run it again.
