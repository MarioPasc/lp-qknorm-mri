"""Dataset converter registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpqknorm.utils.exceptions import ConverterError


if TYPE_CHECKING:
    from lpqknorm.data.converter import DatasetConverter

_REGISTRY: dict[str, type] = {}


def _ensure_loaded() -> None:
    if _REGISTRY:
        return
    from lpqknorm.data.converters.brats_men import BraTSMenConverter

    _REGISTRY["brats_men"] = BraTSMenConverter


def get_converter(name: str) -> DatasetConverter:
    """Instantiate a converter by name.

    Parameters
    ----------
    name : str
        Converter name (e.g. ``"brats_men"``).

    Returns
    -------
    DatasetConverter
        Instantiated converter.

    Raises
    ------
    ConverterError
        If the name is not in the registry.
    """
    _ensure_loaded()
    if name not in _REGISTRY:
        raise ConverterError(
            f"Unknown converter '{name}'",
            {"available": sorted(_REGISTRY.keys())},
        )
    return _REGISTRY[name]()  # type: ignore[no-any-return]
