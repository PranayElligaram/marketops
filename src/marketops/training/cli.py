"""CLI: `python -m marketops.training.cli [--data data/raw] [--out artifacts/models]`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from marketops.data.store import BENCHMARK, DEFAULT_UNIVERSE, fingerprint, load_symbol
from marketops.features.build import build_dataset
from marketops.training.pipeline import run


def load_dataset(root: Path, symbols: list[str]) -> pd.DataFrame:
    frames = {s: load_symbol(root, s) for s in symbols + [BENCHMARK]}
    # Drop a possibly incomplete current-day bar.
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    return build_dataset(frames, BENCHMARK, drop_after=cutoff)


def post_train_reports(ds: pd.DataFrame, model_dir: Path) -> None:
    """SHAP importance (test sample) and drift report (last 60 sessions vs development)."""
    import joblib

    from marketops.explainability.shap_explain import global_importance
    from marketops.features.build import FEATURES
    from marketops.monitoring.checks import drift_report
    from marketops.training.pipeline import DEV_END, TEST_START, labelled

    model = joblib.load(model_dir / "model.joblib")
    dev = labelled(ds[ds["date"] <= DEV_END])
    test = labelled(ds[ds["date"] >= TEST_START])
    imp = global_importance(
        model,
        dev[FEATURES].sample(500, random_state=0),
        test[FEATURES].sample(1000, random_state=0),
    )
    (model_dir / "shap.json").write_text(json.dumps({"mean_abs_shap_logodds": imp}, indent=2))
    last = sorted(ds["date"].unique())[-60:]
    recent = ds[ds["date"].isin(last)]
    ref_p = model.predict_proba(dev[FEATURES])[:, 1]
    new_p = model.predict_proba(recent[FEATURES])[:, 1]
    rep = drift_report(dev, recent, ref_p, new_p)
    rep["recent_window"] = [str(min(last).date()), str(max(last).date())]
    (model_dir / "monitoring.json").write_text(json.dumps(rep, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marketops-train")
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/models"))
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--mlflow", action="store_true", help="log the run to local MLflow")
    args = ap.parse_args(argv)
    ds = load_dataset(args.data, [s.upper() for s in args.symbols])
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Content hashes of the exact CSVs used, independent of the (mutable) ingest manifest.
    manifest["fingerprint"] = fingerprint(
        args.data, [*(s.upper() for s in args.symbols), BENCHMARK]
    )
    res = run(ds, args.out, manifest)
    s = res.summary
    post_train_reports(ds, res.model_dir)
    if args.mlflow:
        from marketops.training.tracking import log_run

        run_id = log_run(s, res.model_dir, Path("artifacts/mlflow.db"))
        print(
            f"mlflow run: {run_id} (view: mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db)"
        )
    print(f"model dir: {res.model_dir}\nselected: {s['selected_model']}  thr={s['threshold']:.3f}")
    print(json.dumps({"cv": s["walk_forward_cv"], "test": s["test"]["model"]}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
