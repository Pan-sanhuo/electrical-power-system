from numpy import array


def case3_repairable():
    """A repairable case: missing REF and abnormal voltage/angle initial values."""
    ppc = {}
    ppc["version"] = "2"
    ppc["baseMVA"] = 100.0
    ppc["bus"] = array(
        [
            [1, 2, 0, 0, 0, 0, 1, 2.0, 220, 110, 1, 1.06, 0.94],
            [2, 2, 20, 10, 0, 0, 1, 1.02, 0, 110, 1, 1.06, 0.94],
            [3, 1, 55, 25, 0, 0, 1, 1.00, 0, 110, 1, 1.06, 0.94],
        ],
        dtype=float,
    )
    ppc["gen"] = array(
        [
            [1, 45, 0, 100, -100, 1.03, 100, 1, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 35, 0, 80, -80, 1.02, 100, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    ppc["branch"] = array(
        [
            [1, 2, 0.02, 0.06, 0.03, 200, 200, 200, 0, 0, 1, -360, 360],
            [1, 3, 0.04, 0.12, 0.02, 150, 150, 150, 0, 0, 1, -360, 360],
            [2, 3, 0.03, 0.10, 0.02, 150, 150, 150, 0, 0, 1, -360, 360],
        ],
        dtype=float,
    )
    return ppc
