from typing import TypeVar, Any
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

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
        """
        target_cfg = self.get_value(
            key=key, 
            is_empty_allowed=is_empty_allowed
        )
        return DataclassInitializer.build(
            cls_=cls_, 
            cfg=target_cfg
            )
    
    def get_value(
        self,
        key: str,
        is_empty_allowed: bool = False
        ) -> Any:
        """
        Retrieve a sub-configuration using a key path.

        Parameters:
        ----------
        key: str
            The dot-notation key path (e.g., "model.params").
        is_empty_allowed: bool
            If True, returns an empty DictConfig instead of raising an error when the key is missing.

        Returns:
        ----------
        Any
            The value of the key.
        
        Raises:
        - ValueError: If the key is not found and is_empty_allowed is False.
        """
        # OmegaConf.select returns None if key is missing (unless default is provided)
        sub_cfg = OmegaConf.select(self.cfg, key)

        if sub_cfg is None:
            if is_empty_allowed:
                # Return an empty DictConfig to maintain type consistency (not a raw dict)
                return OmegaConf.create({})
            raise ValueError(f"Configuration key '{key}' not found in the provided config.")
        return sub_cfg