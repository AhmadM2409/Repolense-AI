import math

import pytest

from core.config import Settings
from tests.retrieval_quality import metrics


@pytest.mark.parametrize(
    "paths,gold,recall,mrr",
    [
        ([], ["a"], 0, 0),
        (["a"], ["a"], 1, 1),
        (["x", "a"], ["a"], 1, 0.5),
        (["a"], ["a", "b"], 0.5, 1),
    ],
)
def test_binary_gold_metrics(paths, gold, recall, mrr):
    result = metrics(paths, gold)
    assert result["recall_at_5"] == recall and result["mrr_at_10"] == mrr
    assert 0 <= result["ndcg_at_10"] <= 1


def test_no_answer_is_not_fabricated_perfect_recall():
    assert metrics(["irrelevant"], []) is None
    assert metrics(["a"], ["a"])["ndcg_at_10"] == 1
    assert metrics(["x", "a"], ["a"])["ndcg_at_10"] == 1 / math.log2(3)
    with pytest.raises(ValueError):
        metrics(["a", "a"], ["a"])


@pytest.mark.parametrize("environment", ["production", "staging", "development", "test"])
def test_settings_configdict_preserves_environment_and_secrets(monkeypatch, environment):
    monkeypatch.setenv("APP_ENV", environment)
    key = "X" * 48
    monkeypatch.setenv("SECRET_KEY", key)
    value = Settings(_env_file=None)
    assert value.app_env == environment and value.secret_key == key
    assert value.model_config["case_sensitive"] is False
    assert value.model_config["env_file"] == ".env"
