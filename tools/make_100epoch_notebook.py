"""Build SAGE_100epochs.ipynb from SAGE_reproduction.ipynb.

The copy is identical except for one extra setup cell that sets the paper's 100 self-retrieval epochs and a
separate Google Drive folder (sage_work_100), so the long run never touches the shorter run's checkpoints or results.
Run `python tools/make_100epoch_notebook.py` after changing SAGE_reproduction.ipynb.
"""
import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = json.load(open(os.path.join(ROOT, "SAGE_reproduction.ipynb"), encoding="utf-8"))
nb = copy.deepcopy(src)

NOTE = {"cell_type": "markdown", "metadata": {}, "source": (
    "> **100-epoch version.** This notebook trains SAGE's self-retrieval model for the paper's full **100 epochs** "
    "(Table B.11), with every other setting as in `SAGE_reproduction.ipynb`. Everything is saved in a separate Drive "
    "folder, `MyDrive/sage_work_100`, so the shorter run in `sage_work` is never changed.\n>\n"
    "> On a free Colab T4 one epoch takes about 55 minutes, so the self-retrieval stage alone needs about 90 GPU hours, "
    "spread over many sessions. Each epoch is saved, so after a disconnect run **Resume** and then the first cell of "
    "section 6 again: it continues from the last finished epoch.")}
SETUP = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": (
    "# [setup] Re-run by the Resume cell at the top after a disconnect.\n"
    "# 100-epoch version: the paper's full self-retrieval schedule, in its own Drive folder.\n"
    "import os\n"
    "os.environ[\"SAGE_SELFRET_EPOCHS\"] = \"100\"\n"
    "os.environ[\"SAGE_DRIVE_DIR\"] = \"sage_work_100\"\n"
    "print(\"100-epoch run: saving to MyDrive/sage_work_100\")")}

first_setup = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and c["source"].startswith("# [setup]"))
nb["cells"].insert(first_setup, SETUP)
nb["cells"].insert(1, NOTE)
with open(os.path.join(ROOT, "SAGE_100epochs.ipynb"), "w", encoding="utf-8") as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=True) + "\n")
print("wrote SAGE_100epochs.ipynb")
