"""Corrected version of case3_bad_data.py.

This case is used to demonstrate the final step after LLM/rule diagnosis:
turn a deliberately flawed 3-bus input file into a physically reasonable
PYPOWER case that can be solved by the power-flow agent.
"""

from numpy import array


def case3_bad_data_fixed():
    """A corrected 3-bus case derived from case3_bad_data.py."""
    ppc = {}
    ppc["version"] = "2"
    ppc["baseMVA"] = 100.0

    # bus columns:
    # BUS_I, BUS_TYPE, PD, QD, GS, BS, AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
    #
    # Corrections versus case3_bad_data.py:
    # 1) bus 2 is set as REF/slack because the only online generator is on bus 2.
    # 2) abnormal initial voltage/angle values are reset near nominal values.
    # 3) voltage limits are kept in the normal order: VMIN < VMAX.
    ppc["bus"] = array(
        [
            [1, 1, 0, 0, 0, 0, 1, 1.00, 0, 110, 1, 1.06, 0.94],
            [2, 3, 40, 20, 0, 0, 1, 1.04, 0, 110, 1, 1.06, 0.94],
            [3, 1, 130, 85, 0, 0, 1, 1.00, 0, 110, 1, 1.06, 0.94],
        ]
    )

    # gen columns:
    # GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    #
    # Corrections versus case3_bad_data.py:
    # 1) QMAX > QMIN.
    # 2) PMAX > PMIN and PMAX is larger than total active load.
    # 3) VG is set to a realistic voltage target.
    ppc["gen"] = array(
        [
            [2, 175, 0, 300, -150, 1.05, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    # branch columns:
    # F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX
    #
    # Corrections versus case3_bad_data.py:
    # 1) branch 1-2 no longer has zero impedance.
    # 2) RATE_A is positive and can be checked as a thermal limit.
    ppc["branch"] = array(
        [
            [1, 2, 0.02, 0.06, 0.03, 250, 250, 250, 0, 0, 1, -360, 360],
            [2, 3, 0.01, 0.04, 0.02, 250, 250, 250, 0, 0, 1, -360, 360],
        ]
    )
    return ppc
