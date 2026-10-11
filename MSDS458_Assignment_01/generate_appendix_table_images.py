import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/msds458-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "assignment_01_outputs"
RESULTS_PATH = OUTPUT_DIR / "experiment_results.csv"

HEADER_COLOR = "#245781"
ALTERNATE_COLOR = "#EAF2F8"
GRID_COLOR = "#C9D1D9"
TEXT_COLOR = "#111111"


def experiment_label(value):
    return value.replace("_", " ")


def draw_table(data, columns, widths, number, title, output_path):
    row_count = len(data)
    figure_height = 1.45 + 0.52 * (row_count + 1)
    fig, ax = plt.subplots(figsize=(16, figure_height))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.text(0.025, 0.955, number, ha="left", va="top", fontsize=16, fontweight="bold", family="DejaVu Serif")
    fig.text(0.025, 0.905, title, ha="left", va="top", fontsize=15, fontstyle="italic", family="DejaVu Serif")

    table = ax.table(
        cellText=data,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        colWidths=widths,
        bbox=[0.025, 0.035, 0.95, 0.81],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.9)
        cell.get_text().set_fontfamily("DejaVu Serif")
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(11)
        else:
            cell.set_facecolor(ALTERNATE_COLOR if row % 2 == 0 else "white")
            cell.get_text().set_color(TEXT_COLOR)
            if column == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.02
            elif number == "Table B1" and column == 1:
                cell.get_text().set_ha("left")
                cell.PAD = 0.02

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


results = pd.read_csv(RESULTS_PATH)

configurations = results.copy()
configurations["Experiment_Number"] = configurations["Experiment"].str.extract(r"E(\d+)").astype(int)
configurations = configurations.sort_values("Experiment_Number")

configuration_rows = []
for row in configurations.itertuples(index=False):
    hidden_layers = "Classifier and three\nregressors" if row.Category == "Stackup" else str(row.Layers).replace("-", " ")
    configuration_rows.append(
        [
            experiment_label(row.Experiment),
            hidden_layers,
            row.Target,
            f"{row.Learning_Rate:.4f}",
            f"{int(row.Batch_Size)}",
            f"{row.Dropout:.2f}",
            f"{int(row.Best_Epoch)}",
        ]
    )

draw_table(
    configuration_rows,
    ["Experiment", "Hidden layers", "Target", "LR", "Batch", "Dropout", "Best epoch"],
    [0.255, 0.17, 0.12, 0.12, 0.12, 0.125, 0.12],
    "Table B1",
    "Neural network and stack up configurations",
    OUTPUT_DIR / "table_B1_model_configurations.png",
)

performance = results.sort_values("Test_MAPE")
performance_rows = []
for row in performance.itertuples(index=False):
    performance_rows.append(
        [
            experiment_label(row.Experiment),
            f"{row.Test_MAE:.2f}",
            f"{row.Test_RMSE:.2f}",
            f"{row.Test_MAPE:.2f}%",
            f"{row.Test_R2:.4f}",
            f"{row.Test_P95_APE:.2f}%",
            f"{row.Test_Within_20_Pct:.2f}%",
        ]
    )

draw_table(
    performance_rows,
    ["Experiment", "MAE", "RMSE", "MAPE", "R squared", "P95 APE", "Within 20%"],
    [0.27, 0.125, 0.125, 0.125, 0.13, 0.13, 0.125],
    "Table B2",
    "Complete test performance by experiment",
    OUTPUT_DIR / "table_B2_test_performance.png",
)
