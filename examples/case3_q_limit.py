from numpy import array


def case3_q_limit():
    """Three-bus example with a tight PV reactive limit.

    It is intentionally small so the agent can demonstrate:
    1. screening without Q-limit enforcement,
    2. detecting PV reactive power limit violation,
    3. rerunning with ENFORCE_Q_LIMS enabled.
    """
    ppc = {}
    ppc["version"] = "2"
    ppc["baseMVA"] = 100.0

    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array(
        [
            [1, 3, 0, 0, 0, 0, 1, 1.04, 0, 110, 1, 1.06, 0.94],
            [2, 2, 30, 15, 0, 0, 1, 1.03, 0, 110, 1, 1.06, 0.94],
            [3, 1, 120, 75, 0, 0, 1, 1.00, 0, 110, 1, 1.06, 0.94],
        ]
    )

    # bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf
    ppc["gen"] = array(
        [
            [1, 80, 0, 80, -80, 1.04, 100, 1, 220, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 80, 0, 12, -8, 1.03, 100, 1, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    # fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
    ppc["branch"] = array(
        [
            [1, 2, 0.020, 0.060, 0.030, 120, 120, 120, 0, 0, 1, -360, 360],
            [1, 3, 0.080, 0.240, 0.025, 90, 90, 90, 0, 0, 1, -360, 360],
            [2, 3, 0.060, 0.180, 0.020, 90, 90, 90, 0, 0, 1, -360, 360],
        ]
    )
    return ppc

