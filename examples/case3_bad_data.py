from numpy import array


def case3_bad_data():
    """A deliberately flawed case for inspect/repair demonstrations."""
    ppc = {}
    ppc["version"] = "2"
    ppc["baseMVA"] = 100.0
    ppc["bus"] = array(
        [
            [1, 1, 0, 0, 0, 0, 1, 2.5, 270, 110, 1, 0.94, 1.06],
            [2, 2, 40, 20, 0, 0, 1, 1.02, 0, 110, 1, 1.06, 0.94],
            [3, 1, 130, 85, 0, 0, 1, 1.00, 0, 110, 1, 1.06, 0.94],
        ]
    )
    ppc["gen"] = array(
        [
            [2, 50, 0, -20, 30, 1.20, 100, 1, 40, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    ppc["branch"] = array(
        [
            [1, 2, 0, 0, 0, -100, 0, 0, 0, 0, 1, -360, 360],
            [2, 3, 0.06, 0.18, 0.02, 90, 90, 90, 0, 0, 1, -360, 360],
        ]
    )
    return ppc

