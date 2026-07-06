import numpy as np


def test_rule_gated_prediction_preserves_baseline_outside_event_mask():
    from analysis.evaluate_rule_gated_ensemble import rule_gated_prediction

    baseline = np.zeros((1, 4, 2), dtype=np.float32)
    event_model = np.full((1, 4, 2), 10.0, dtype=np.float32)
    masks = np.zeros((1, 4, 3, 2), dtype=np.float32)
    masks[:, 1:3, 0, :] = 1.0

    pred = rule_gated_prediction(baseline, event_model, masks)

    np.testing.assert_allclose(pred[:, 0, :], baseline[:, 0, :])
    np.testing.assert_allclose(pred[:, 3, :], baseline[:, 3, :])
    np.testing.assert_allclose(pred[:, 1:3, :], event_model[:, 1:3, :])


def test_rule_gated_prediction_supports_single_channel_mask_broadcast():
    from analysis.evaluate_rule_gated_ensemble import rule_gated_prediction

    baseline = np.zeros((1, 3, 2), dtype=np.float32)
    event_model = np.ones((1, 3, 2), dtype=np.float32)
    mask = np.zeros((1, 3, 1), dtype=np.float32)
    mask[:, 2, :] = 1.0

    pred = rule_gated_prediction(baseline, event_model, mask, alpha=0.5)

    np.testing.assert_allclose(pred[:, :2, :], 0.0)
    np.testing.assert_allclose(pred[:, 2, :], 0.5)
