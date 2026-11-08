from dataclasses import dataclass
from typing import TypeVar, cast

from dataclass_initializer import DataclassInitializer
from omegaconf import DictConfig, ListConfig, OmegaConf

T = TypeVar("T")


@dataclass
class DictConfigHandler:
    """
    Read values from an OmegaConf ``DictConfig`` and build dataclasses from its sections.

    Keeps OmegaConf-specific access out of business logic.

    Attributes
    ----------
    cfg : DictConfig
        The root configuration object.
    """

    cfg: DictConfig

    def build_dataclass(
        self, cls_: type[T], key: str, is_empty_allowed: bool = False
    ) -> T:
        """
        Build a dataclass instance from the section at a key path.

        Parameters
        ----------
        cls_ : type[T]
            The target dataclass type.
        key : str
            The dot-notation key path (e.g., ``"model.params"``).
        is_empty_allowed : bool
            If True, a missing key builds the instance from the dataclass defaults only.

        Returns
        -------
        T
            The built dataclass instance.

        Raises
        ------
        ValueError
            If the key is not found and ``is_empty_allowed`` is False.
        TypeError
            If the value at the key is not a ``DictConfig``.
        """
        section = self.get_value(key=key, is_empty_allowed=is_empty_allowed)
        if not isinstance(section, DictConfig):
            raise TypeError(
                f"Configuration key '{key}' must be a DictConfig, got {type(section).__name__}."
            )
        return DataclassInitializer.build(cls_=cls_, cfg=section)

    def get_value(self, key: str, is_empty_allowed: bool = False) -> object:
        """
        Retrieve a configuration value at a key path.

        A key whose value is ``null`` or the missing marker ``???`` is treated as not found.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., ``"model.params"``).
        is_empty_allowed : bool
            If True, a missing key returns an empty ``DictConfig`` instead of raising.

        Returns
        -------
        object
            The value at the key path (a ``DictConfig`` or ``ListConfig`` sub-tree, or a scalar).

        Raises
        ------
        ValueError
            If the key is not found and ``is_empty_allowed`` is False.
        """
        selected = cast(object, OmegaConf.select(self.cfg, key))
        if selected is not None:
            return selected
        if is_empty_allowed:
            return OmegaConf.create({})
        raise ValueError(f"Configuration key '{key}' not found in the provided config.")

    def get_str_value(self, key: str) -> str:
        """
        Retrieve a string value at a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., ``"model.name"``).

        Returns
        -------
        str
            The string value at the key path.

        Raises
        ------
        ValueError
            If the key is not found.
        TypeError
            If the value is not a string.
        """
        return self._get_typed_value(key=key, expected_type=str)

    def get_int_value(self, key: str) -> int:
        """
        Retrieve an integer value at a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., ``"model.batch_size"``).

        Returns
        -------
        int
            The integer value at the key path.

        Raises
        ------
        ValueError
            If the key is not found.
        TypeError
            If the value is not an integer.
        """
        return self._get_typed_value(key=key, expected_type=int)

    def get_float_value(self, key: str) -> float:
        """
        Retrieve a float value at a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., ``"model.lr"``).

        Returns
        -------
        float
            The float value at the key path.

        Raises
        ------
        ValueError
            If the key is not found.
        TypeError
            If the value is not a float. Integers are not converted.
        """
        return self._get_typed_value(key=key, expected_type=float)

    def get_bool_value(self, key: str) -> bool:
        """
        Retrieve a boolean value at a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., ``"model.is_enabled"``).

        Returns
        -------
        bool
            The boolean value at the key path.

        Raises
        ------
        ValueError
            If the key is not found.
        TypeError
            If the value is not a boolean.
        """
        return self._get_typed_value(key=key, expected_type=bool)

    def get_dictconfig_value(self, key: str) -> DictConfig:
        """
        Retrieve a ``DictConfig`` sub-tree at a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., ``"model.params"``).

        Returns
        -------
        DictConfig
            The sub-tree at the key path.

        Raises
        ------
        ValueError
            If the key is not found.
        TypeError
            If the value is not a ``DictConfig``.
        """
        return self._get_typed_value(key=key, expected_type=DictConfig)

    def get_listconfig_value(self, key: str) -> ListConfig:
        """
        Retrieve a ``ListConfig`` sequence at a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., ``"model.layers"``).

        Returns
        -------
        ListConfig
            The sequence at the key path.

        Raises
        ------
        ValueError
            If the key is not found.
        TypeError
            If the value is not a ``ListConfig``.
        """
        return self._get_typed_value(key=key, expected_type=ListConfig)

    def _get_typed_value(self, key: str, expected_type: type[T]) -> T:
        selected = self.get_value(key=key)
        message = f"Configuration key '{key}' must be {expected_type.__name__}, got {type(selected).__name__}."
        if not isinstance(selected, expected_type):
            raise TypeError(message)
        return selected
