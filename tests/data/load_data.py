import importlib.resources

import numpy as np


def load_data() -> dict[str, np.ndarray]:
    with importlib.resources.open_binary(__package__, "unit_test_data.npz") as file:  # type: ignore
        obj = np.load(file)
        data = {key: obj[key] for key in obj.files}
    return data
