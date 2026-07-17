import os


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)
    parser.addoption("--zind-root", default=os.environ.get("ZIND_ROOT"))
    parser.addoption("--heldout", default=os.environ.get("VAL_HOMES"))
    parser.addoption("--ckpt", default=os.environ.get("CKPT", "weights/best.pt"))
