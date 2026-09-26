import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.evaluation.evaluate import derived_binary
from src.evaluation.metrics import (
    classification_metrics,
    confusion,
    expected_calibration_error,
    top_confusions,
    top_k_accuracy,
)


def test_multiclass_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 500)
    y_pred = np.where(rng.random(500) < 0.8, y_true, rng.integers(0, 4, 500))
    probs = rng.dirichlet(np.ones(4), 500)
    m = classification_metrics(y_true, y_pred, ["a", "b", "c", "d"], probs=probs)

    assert m["accuracy"] == pytest.approx((y_true == y_pred).mean())
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y_true, y_pred, average="weighted"))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert sum(row["support"] for row in m["per_class"]) == 500
    assert "top3_accuracy" in m and "roc_auc" not in m


def test_binary_metrics_report_sensitivity_and_specificity():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0])
    probs = np.stack([1 - y_pred * 0.9, y_pred * 0.9], axis=1)
    m = classification_metrics(y_true, y_pred, ["healthy", "diseased"], probs=probs)
    assert m["sensitivity"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["false_negatives"] == 1 and m["false_positives"] == 1
    assert 0 <= m["roc_auc"] <= 1


def test_ece_is_zero_for_perfectly_calibrated_confident_model():
    y = np.array([0, 1, 2, 1])
    probs = np.eye(3)[y]
    assert expected_calibration_error(probs, y) == pytest.approx(0.0)
    assert expected_calibration_error(np.eye(3)[[1, 2, 0, 0]], y) == pytest.approx(1.0)


def test_top_k_accuracy():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    y = np.array([1, 0])
    assert top_k_accuracy(probs, y, 1) == 0.0
    assert top_k_accuracy(probs, y, 2) == 0.5
    assert top_k_accuracy(probs, y, 3) == 1.0


def test_top_confusions_orders_off_diagonal_cells():
    cm = confusion([0, 0, 0, 1, 1, 2], [1, 1, 0, 2, 1, 2], 3)
    rows = top_confusions(cm, ["a", "b", "c"])
    assert rows[0] == {"true": "a", "predicted": "b", "count": 2, "share_of_true_class": pytest.approx(2 / 3)}
    assert len(rows) == 2


def test_derived_binary_sums_disease_probabilities():
    class_names = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy"]
    probs = np.array([
        [0.3, 0.3, 0.4],   # most likely healthy, but diseased mass is 0.6
        [0.0, 0.1, 0.9],
    ])
    m = derived_binary(np.array([0, 2]), probs, class_names)
    assert m["accuracy"] == 1.0
