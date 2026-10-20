
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "reports" / "training_history.json"
OUTPUT_PATH = ROOT / "reports" / "figures" / "training_curves.png"


def main():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"{HISTORY_PATH} not found. Train the model first."
        )

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    epochs = [item["epoch"] for item in history]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Loss", "Validation metrics"),
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[item["train_loss"] for item in history],
            mode="lines+markers",
            name="Train loss",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[item["val_loss"] for item in history],
            mode="lines+markers",
            name="Validation loss",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[item["val_mean_iou"] for item in history],
            mode="lines+markers",
            name="Validation mIoU",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[item["val_mean_dice"] for item in history],
            mode="lines+markers",
            name="Validation Dice",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="MobMask — Training History",
        template="plotly_white",
        height=520,
        width=1200,
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 1], row=1, col=2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUTPUT_PATH), scale=2)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
