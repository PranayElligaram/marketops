"""Optional local MLflow tracking (SQLite backend, no server, no cost)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def log_run(summary: dict[str, Any], model_dir: Path, tracking_db: Path) -> str:
    import mlflow

    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve().as_posix()}")
    name = "marketops-5d-direction"
    if mlflow.get_experiment_by_name(name) is None:
        # Default artifact root is ./mlruns relative to the CWD, which litters whichever
        # directory the process happens to run in; pin it beside the tracking DB.
        mlflow.create_experiment(
            name, artifact_location=(tracking_db.resolve().parent / "mlartifacts").as_uri()
        )
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=summary["version"]) as run:
        mlflow.log_params(
            {
                "selected_model": summary["selected_model"],
                "threshold": round(summary["threshold"], 6),
                "dev_end": summary["dev_end"],
                "test_start": summary["test_start"],
                "n_symbols": len(summary["symbols"]),
                "train_rows": summary["train_rows_final"],
                "model_version": summary["version"],
                **{f"env_{k}": v for k, v in summary["env"].items()},
            }
        )
        for name, cv in summary["walk_forward_cv"].items():
            mlflow.log_metrics({f"cv.{name}.{k}": v for k, v in cv.items()})
        t = summary["test"]["model"]
        mlflow.log_metrics(
            {
                f"test.{k}": t[k]
                for k in ("roc_auc", "pr_auc", "brier", "log_loss", "precision", "recall", "f1")
                if t[k] is not None
            }
        )
        mlflow.log_metrics({f"latency.{k}": v for k, v in summary["latency"].items()})
        for f in ("metadata.json", "model.joblib"):
            mlflow.log_artifact(str(model_dir / f))
        return str(run.info.run_id)
