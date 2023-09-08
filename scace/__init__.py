from .scace_train import run_scace
import importlib

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

__name__ = "scAce"
try:
    __version__ = version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = '0.1.2'