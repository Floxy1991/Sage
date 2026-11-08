#!/usr/bin/env bash
# Full SAGE reproduction with the paper's settings (Tables B.11, B.12) on a Linux GPU server (e.g. a DGX).
# Start it in the background so it survives closing the browser:   nohup bash run_paper_gpu.sh > train_log.txt 2>&1 &
# Watch it with:   tail -f train_log.txt
# Safe to re-run: every stage continues where it stopped and finished stages are skipped. See DGX.md.
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
EPOCHS="${SAGE_SELFRET_EPOCHS:-100}"

python -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NONE - see DGX.md'); sys.exit(0 if ok else 1)"

echo "=== 1/5 Self-retrieval LM: $EPOCHS epochs (lr 1e-4, batch 16 x 8 = 128, 1000 warmup steps, linear) ==="
python sage.py train-selfret --epochs "$EPOCHS" --lr 1e-4 --bs 16 --grad-accum 8 --warmup 1000 --sched linear --out ckpt/selfret

echo "=== 2/5 ResID index: RQ-VAE, 50 epochs ==="
if [ -f ckpt/index/index.pt ]; then echo "ckpt/index exists: skipping"; else
    python sage.py build-index --model ckpt/selfret --epochs 50 --out ckpt/index; fi

echo "=== 3/5 Fusion generator: 3 epochs (lr 2e-5, batch 32, cosine) ==="
python sage.py train-fusion --selfret ckpt/selfret --index ckpt/index --knowledge none --epochs 3 --lr 2e-5 --bs 32 \
    --micro-bs 4 --warmup 0.03 --sched cosine --out ckpt/fusion

echo "=== 4/5 Inference on the whole test set (paper decoding) ==="
python sage.py infer --selfret ckpt/selfret --index ckpt/index --fusion ckpt/fusion --knowledge none --sample \
    --out results/sage_paper100

echo "=== 5/5 Retrieval evaluation ==="
python sage.py eval-retrieval --selfret ckpt/selfret --index ckpt/index --out results/sage_paper100

echo "ALL DONE. Results are in results/sage_paper100/metrics.json"
python compare.py
