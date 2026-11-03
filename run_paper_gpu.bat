@echo off
REM Full SAGE reproduction with the paper's settings (Tables B.11, B.12) on a Windows PC with an NVIDIA GPU.
REM Double-click it, or run it from Command Prompt in this folder. Safe to re-run: every stage continues
REM where it stopped (training from the last finished epoch, inference from the last saved example),
REM and finished stages are skipped. See SCHOOL_GPU.md.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONUTF8=1

python -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NONE - install the CUDA build of PyTorch, see SCHOOL_GPU.md'); sys.exit(0 if ok else 1)"
if errorlevel 1 goto fail

echo.
echo === 1/5 Self-retrieval LM: 100 epochs (lr 1e-4, batch 16 x 8 = 128, 1000 warmup steps, linear) ===
python sage.py train-selfret --epochs 100 --lr 1e-4 --bs 16 --grad-accum 8 --warmup 1000 --sched linear --out ckpt/selfret
if errorlevel 1 goto fail

echo.
echo === 2/5 ResID index: RQ-VAE, 50 epochs ===
if exist ckpt\index\index.pt (
    echo ckpt\index exists: skipping
) else (
    python sage.py build-index --model ckpt/selfret --epochs 50 --out ckpt/index
    if errorlevel 1 goto fail
)

echo.
echo === 3/5 Fusion generator: 3 epochs (lr 2e-5, batch 32, cosine) ===
python sage.py train-fusion --selfret ckpt/selfret --index ckpt/index --knowledge none --epochs 3 --lr 2e-5 --bs 32 --micro-bs 4 --warmup 0.03 --sched cosine --out ckpt/fusion
if errorlevel 1 goto fail

echo.
echo === 4/5 Inference on the whole test set (paper decoding) ===
python sage.py infer --selfret ckpt/selfret --index ckpt/index --fusion ckpt/fusion --knowledge none --sample --out results/sage_paper100
if errorlevel 1 goto fail

echo.
echo === 5/5 Retrieval evaluation ===
python sage.py eval-retrieval --selfret ckpt/selfret --index ckpt/index --out results/sage_paper100
if errorlevel 1 goto fail

echo.
echo ALL DONE. Results are in results\sage_paper100\metrics.json
python compare.py
pause
exit /b 0

:fail
echo.
echo STOPPED WITH AN ERROR (see above). Fix it, or send a screenshot, then run this file again: it continues where it stopped.
pause
exit /b 1
