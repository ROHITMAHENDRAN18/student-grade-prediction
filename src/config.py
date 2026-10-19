"""Project-wide configuration: paths, random seed, column groups and plot style."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # headless backend (no display needed)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student-por.csv"
EDA_DIR = ROOT / "outputs" / "eda"
EVAL_DIR = ROOT / "outputs" / "model_evaluation"
MODEL_DIR = ROOT / "models"
for d in (EDA_DIR, EVAL_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "G3"                               # final grade, 0-20 (regression target)
PASS_MARK = 10                              # G3 >= 10 counts as "pass" in the Portuguese system
GRADE_COLS = ["G1", "G2"]                   # earlier period grades

# ---- Column groups (32 input features) --------------------------------------
BINARY_COLS = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid",
               "activities", "nursery", "higher", "internet", "romantic"]
NOMINAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]
# Everything else is numeric: ordinal 1-5 ratings and counts are kept as numbers.
ORDINAL_COLS = ["Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
                "goout", "Dalc", "Walc", "health"]

# ---- Plot style: validated categorical slots 1-3 + neutral inks --------------
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PASS_COLORS = {"Pass": BLUE, "Fail": ORANGE}

def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

def save_fig(fig, path):
    fig.savefig(path)
    plt.close(fig)
