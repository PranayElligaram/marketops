"""Research API tests using a tiny model trained on synthetic data in tmp_path."""

import hashlib
import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from api.main import app
from marketops.config import get_settings
from marketops.data.store import cache_path
from marketops.features.build import FEATURES, build_dataset
from marketops.training import pipeline
from tests.test_ml_backtest import synth


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data, models = tmp_path / "raw", tmp_path / "models"
    data.mkdir()
    frames = {"AAA": synth(seed=1), "SPY": synth(seed=2)}
    for s, f in frames.items():
        f.to_csv(cache_path(data, s), index=False, date_format="%Y-%m-%d")
    ds = build_dataset(frames).dropna(subset=["label"])
    model = pipeline.make_logreg()
    model.fit(ds[FEATURES], ds["label"])
    v = models / "v1"
    v.mkdir(parents=True)
    joblib.dump(model, v / "model.joblib")
    meta = {
        "version": "v1",
        "selected_model": "logreg",
        "threshold": 0.5,
        "model_sha256": hashlib.sha256((v / "model.joblib").read_bytes()).hexdigest(),
        "symbols": ["AAA"],
        "features": FEATURES,
        "walk_forward_cv": {},
        "dev_baselines": {},
        "test": {},
        "test_other_candidates_not_for_selection": {},
        "caveats": [],
    }
    (v / "metadata.json").write_text(json.dumps(meta))
    (v / "predictions.csv").write_text(
        "date,symbol,label,fwd_ret,prob,split\n2024-01-02,AAA,1,0.01,0.6,test\n"
    )
    (models / "LATEST").write_text("v1")
    monkeypatch.setenv("MARKETOPS_DATA_DIR", str(data))
    monkeypatch.setenv("MARKETOPS_MODEL_DIR", str(models))
    get_settings.cache_clear()
    yield models
    get_settings.cache_clear()


def test_no_model_gives_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETOPS_MODEL_DIR", str(tmp_path))
    get_settings.cache_clear()
    with TestClient(app) as c:
        assert c.get("/model/metadata").status_code == 503
    get_settings.cache_clear()


def test_analysis_history_and_errors(env: Path) -> None:
    with TestClient(app) as c:
        r = c.get("/analysis/aaa")
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["probability_up"] <= 1 and body["persisted"] is False
        assert set(body["features"]) == set(FEATURES)
        assert "not investment advice" in body["disclaimer"]
        assert c.get("/analysis/ZZZ").status_code == 404
        assert c.get("/analysis/SPY").status_code == 404
        h = c.get("/market/AAA/history?days=10").json()
        assert len(h["rows"]) == 10 and h["rows"][0]["date"] < h["rows"][-1]["date"]
        assert c.get("/market/AAA/history?days=1").status_code == 422
        assert c.get("/predictions/AAA").json()[0]["probability_up"] == 0.6
        assert c.get("/backtest").status_code == 503
        assert c.get("/model/evaluation").status_code == 200
