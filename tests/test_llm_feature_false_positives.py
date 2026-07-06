import numpy as np


def test_binary_feature_false_positive_metrics():
    from analysis.diagnose_llm_feature_false_positives import binary_confusion_metrics

    predicted = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    actual = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)

    metrics = binary_confusion_metrics(predicted, actual)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 2
    assert metrics["fn"] == 1
    assert metrics["tn"] == 2
    assert metrics["precision"] == 1 / 3
    assert metrics["recall"] == 1 / 2
    assert metrics["false_positive_ratio_among_predicted"] == 2 / 3
