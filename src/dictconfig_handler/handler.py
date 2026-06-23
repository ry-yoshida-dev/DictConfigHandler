from typing import TypeVar, cast
from dataclasses import dataclass
from omegaconf import DictConfig, ListConfig, OmegaConf

from dataclass_initializer import DataclassInitializer

T = TypeVar('T')

@dataclass
class DictConfigHandler:
    """
    DictConfigHandler is a utility class for handling DictConfig operations.
    It isolates OmegaConf dependency logic from the business logic.

    Attributes:
    ----------
    cfg: DictConfig
        The root configuration object.
    """
    cfg: DictConfig

    def build_dataclass(
        self, 
        cls_: type[T],
        key: str,
        is_empty_allowed: bool = False
        ) -> T:
        """
        Build a dataclass instance from a specific section of the DictConfig.

        Parameters:
        ----------
        cls_: type[T]
            The target dataclass type.
        key: str
            The dot-notation key path (e.g., "model.params").
        is_empty_allowed: bool
            If True, returns an empty config if the key is missing.

        Returns:
        ----------
        T
            The instantiated dataclass.

        Raises:
        ------
        ValueError
            If the key is not found and is_empty_allowed is False.
        TypeError
            If the value at the key is not a DictConfig.
        """
        if is_empty_allowed:
            target_cfg = self.get_value(key=key, is_empty_allowed=True)
            if not isinstance(target_cfg, DictConfig):
                raise TypeError(
                    f"Configuration key '{key}' must be a DictConfig, "
                    + f"got {type(target_cfg).__name__}."
                )
        else:
            target_cfg = self.get_dictconfig_value(key=key)
        return DataclassInitializer.build(
            cls_=cls_,
            cfg=target_cfg,
        )
    
    def get_value(
        self,
        key: str,
        is_empty_allowed: bool = False
        ) -> object:
        """
        Retrieve a configuration value using a key path.

        Parameters:
        ----------
        key: str
            The dot-notation key path (e.g., "model.params").
        is_empty_allowed: bool
            If True, returns an empty DictConfig instead of raising an error when the key is missing.

        Returns:
        ----------
        object
            The value at the key path (typically a DictConfig sub-tree or a scalar).

        Raises:
        ------
        ValueError
            If the key is not found and is_empty_allowed is False.
        """
        # OmegaConf.select returns None if key is missing (unless default is provided)
        selected = OmegaConf.select(self.cfg, key)

        if selected is None:
            if is_empty_allowed:
                # Return an empty DictConfig to maintain type consistency (not a raw dict)
                return OmegaConf.create({})
            raise ValueError(f"Configuration key '{key}' not found in the provided config.")
        return cast(object, selected)

    def _get_typed_value(
        self,
        key: str,
        expected_type: type[T],
        type_label: str,
    ) -> T:
        """
        Retrieve a configuration value and validate its runtime type.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., "model.params").
        expected_type : type[T]
            The expected Python type of the value.
        type_label : str
            Human-readable type name used in error messages.

        Returns
        -------
        T
            The value at the key path, validated against ``expected_type``.

        Raises
        ------
        ValueError
            If the key is not found.
        TypeError
            If the value at the key does not match ``expected_type``.
        """
        selected = OmegaConf.select(self.cfg, key)
        if selected is None:
            raise ValueError(
                f"Configuration key '{key}' not found in the provided config."
            )
        if not isinstance(selected, expected_type):
            raise TypeError(
                f"Configuration key '{key}' must be a {type_label}, "
                + f"got {type(selected).__name__}."
            )
        return selected

    def get_str_value(self, key: str) -> str:
        """
        Retrieve a string value using a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., "model.name").

        Returns
        -------
        str
            The string value at the key path.
        """
        return self._get_typed_value(key, str, "string")

    def get_int_value(self, key: str) -> int:
        """
        Retrieve an integer value using a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., "model.batch_size").

        Returns
        -------
        int
            The integer value at the key path.
        """
        return self._get_typed_value(key, int, "integer")

    def get_float_value(self, key: str) -> float:
        """
        Retrieve a float value using a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., "model.lr").

        Returns
        -------
        float
            The float value at the key path.
        """
        return self._get_typed_value(key, float, "float")

    def get_bool_value(self, key: str) -> bool:
        """
        Retrieve a boolean value using a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., "model.enabled").

        Returns
        -------
        bool
            The boolean value at the key path.
        """
        return self._get_typed_value(key, bool, "boolean")

    def get_dictconfig_value(self, key: str) -> DictConfig:
        """
        Retrieve a DictConfig sub-tree using a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., "model.params").

        Returns
        -------
        DictConfig
            The DictConfig sub-tree at the key path.
        """
        return self._get_typed_value(key, DictConfig, "DictConfig")

    def get_listconfig_value(self, key: str) -> ListConfig:
        """
        Retrieve a ListConfig sequence using a key path.

        Parameters
        ----------
        key : str
            The dot-notation key path (e.g., "model.layers").

        Returns
        -------
        ListConfig
            The ListConfig sequence at the key path.
        """
        return self._get_typed_value(key, ListConfig, "ListConfig")
