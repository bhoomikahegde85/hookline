import numpy as np

from hookline import brain_networks as bn


def test_atlas_covers_both_hemispheres_with_seven_networks():
    nets = bn.vertex_networks()
    assert nets.shape == (2 * bn.N_PER_HEMI,)
    assert set(np.unique(nets)) == {-1, *range(len(bn.NETWORKS))}


def test_network_curves_recover_a_planted_network():
    nets = bn.vertex_networks()
    preds = np.zeros((6, len(nets)))
    preds[:, nets == bn.NETWORKS.index("Vis")] = np.arange(6)[:, None]
    c = bn.network_curves(preds)
    assert np.allclose(c["Vis"], np.arange(6)) and np.allclose(c["Default"], 0)


def test_features_and_peaks():
    nets = bn.vertex_networks()
    preds = np.zeros((8, len(nets)))
    preds[3, nets == bn.NETWORKS.index("DorsAttn")] = 2.0
    f = bn.network_features(preds, list(range(8)))
    assert f["peak_time_s"] == 3.0 and f["peak_height"] > 1
    assert bn.network_features(preds[:3], [0, 1, 2]) is None


def test_dominant_network_reading_flags_visual_tracking():
    curves = {n: np.ones(10) for n in bn.NETWORKS}
    curves["Vis"] *= 31
    curves["DorsAttn"] *= 32
    curves["Limbic"] *= 3
    curves["Cont"] *= 5
    curves["Default"] *= 6
    curves["SomMot"] *= 12
    curves["SalVentAttn"] *= 11
    shares = bn.network_activation_shares(curves, list(range(10)))
    reading = bn.network_reading(shares)
    assert [x["network"] for x in reading["top"]] == ["DorsAttn", "Vis"]
    assert reading["emotional_signal"]["level"] == "low"
    assert "visual tracking" in reading["summary"]
    assert "verbal prompt" in reading["recommendation"]
