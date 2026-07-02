# tests/test_device.py

from mlcore.training.utils import get_device


def test_get_device_auto():
    device = get_device("auto")
    assert device.type in ["cpu", "cuda", "mps"]
