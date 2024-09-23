import numpy as np
from importlib import resources

DATA_MODULE = "data"


def load_cats(size:int = 48) -> tuple:
    cats_path = resources.files(DATA_MODULE) / "cats/cats_{}.csv".format(size)
    data = np.genfromtxt(cats_path, dtype=np.uint8)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y

def load_age(size:int = 48) -> tuple:
    age_path = resources.files(DATA_MODULE) / "age/age_{}.csv".format(size)
    data = np.genfromtxt(age_path, dtype=np.uint8)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


__all__ = [
    'load_cats',
    'load_age'
]