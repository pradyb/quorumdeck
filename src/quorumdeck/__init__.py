"""quorumdeck -- a terminal console for running several AI agents side by side."""

from importlib.metadata import PackageNotFoundError, version

try:
    # pyproject.toml's `version` is the one source of truth; a hardcoded
    # string here is exactly the kind of thing a release bumps in one place
    # and forgets in the other.
    __version__ = version("quorumdeck")
except PackageNotFoundError:  # pragma: no cover - only an uninstalled checkout
    __version__ = "0+unknown"

__all__ = ["__version__"]
