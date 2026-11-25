import numpy as np

from certalarm import certify as C, decision as D, sim as S


def test_unit_shapes_and_labels():
    u = S.simulate_unit(np.random.default_rng(0))
    T = len(u["rul"])
    assert T >= 20 and u["mu"].shape == (T,) and u["sigma"].shape == (T,)
    assert u["rul"][0] == 1.0 and u["rul"][-1] == 0.0
    assert u["mu"].min() >= 0.0 and u["mu"].max() <= 1.0


def test_fleet_is_heterogeneous():
    lives = [len(u["rul"]) for u in S.simulate_fleet(50, np.random.default_rng(1))]
    assert np.std(lives) > 20


def test_crc_controls_realized_risk_on_synthetic_fleet():
    alpha, lead, n, draws = 0.1, 0.05, 19, 600
    thetas = np.round(np.arange(0.0, 1.02, 0.01), 2)
    late = []
    for d in range(draws):
        fleet = S.simulate_fleet(n + 1, np.random.default_rng([7, d]))
        Lm = np.stack([D.outcome_curves(u["mu"], u["rul"], thetas, lead)[0] for u in fleet[:n]])
        th = C.crc_threshold(Lm, thetas, alpha)
        assert th is not None
        t = fleet[n]
        late.append(D.alarm_outcome(t["mu"], t["rul"], th, lead)[0])
    assert np.mean(late) <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / draws)
