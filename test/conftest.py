import torch


def pytest_configure(config):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
