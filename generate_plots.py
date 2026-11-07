"""Plot recorded training rewards and KB sizes. Missing measurements are errors.
No trained-versus-baseline comparison is inferred from a training log.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def reward_points(metrics):
    points = [(row["step"], row.get("reward", row.get("reward/mean")))
              for row in metrics.get("log_history", [])
              if "step" in row and ("reward" in row or "reward/mean" in row)]
    if not points:
        raise ValueError("No recorded rewards; refusing to synthesize a training curve")
    return points

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = json.loads((ROOT / "results/training_metrics.json").read_text())
    baseline = json.loads((ROOT / "results/baseline_results.json").read_text())
    points = reward_points(metrics)
    if not baseline:
        raise ValueError("No recorded baseline episodes")
    output = ROOT / "plots"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(*zip(*points))
    ax.set(xlabel="Recorded training step", ylabel="Reward", title="Recorded training rewards")
    fig.tight_layout()
    fig.savefig(output / "reward_curve.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([r["episode"] for r in baseline], [r["kb_size"] for r in baseline], marker="o")
    ax.set(xlabel="Episode", ylabel="Entries", title="Knowledge base in recorded baseline episodes")
    fig.tight_layout()
    fig.savefig(output / "kb_growth.png")
    plt.close(fig)
    print("Saved recorded reward and KB plots. No held-out model comparison is available.")

if __name__ == "__main__":
    main()
