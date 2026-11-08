"""Tests for DictConfigHandler (get_value, build_dataclass)."""
import pytest
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

from dictconfig_handler import DictConfigHandler


# --- Sample dataclasses for testing ---
@dataclass
class ModelParams:
    """Sample class for build_dataclass tests."""
    lr: float = 0.001
    batch_size: int = 32
    name: str = "default"


@dataclass
class NestedConfig:
    """Sample class for nested key tests."""
    depth: int = 1
    width: int = 64


# --- get_value tests ---
class TestGetValue:
    def test_get_value_exists(self):
        cfg = OmegaConf.create({
            "model": {"lr": 0.01, "batch_size": 64},
            "data": {"path": "/tmp"},
        })
        handler = DictConfigHandler(cfg=cfg)
        sub = handler.get_value("model")
        assert isinstance(sub, DictConfig)
        assert sub["lr"] == 0.01
        assert sub["batch_size"] == 64

    def test_get_value_nested_key(self):
        cfg = OmegaConf.create({
            "a": {"b": {"c": 42}},
        })
        handler = DictConfigHandler(cfg=cfg)
        sub = handler.get_value("a.b")
        assert isinstance(sub, DictConfig)
        assert sub["c"] == 42

    def test_get_value_missing_raises(self):
        cfg = OmegaConf.create({"foo": 1})
        handler = DictConfigHandler(cfg=cfg)
        with pytest.raises(ValueError, match="Configuration key 'missing' not found"):
            handler.get_value("missing")

    def test_get_value_missing_with_empty_allowed(self):
        cfg = OmegaConf.create({"foo": 1})
        handler = DictConfigHandler(cfg=cfg)
        sub = handler.get_value("missing", is_empty_allowed=True)
        assert isinstance(sub, DictConfig)
        assert sub == OmegaConf.create({})
        assert len(sub) == 0


# --- build_dataclass tests ---
class TestBuildDataclass:
    def test_build_dataclass_from_key(self):
        cfg = OmegaConf.create({
            "model": {
                "lr": 0.01,
                "batch_size": 64,
                "name": "resnet",
            },
        })
        handler = DictConfigHandler(cfg=cfg)
        params = handler.build_dataclass(ModelParams, key="model")
        assert params.lr == 0.01
        assert params.batch_size == 64
        assert params.name == "resnet"

    def test_build_dataclass_partial_uses_defaults(self):
        """When only some keys are present, defaults are used for the rest."""
        cfg = OmegaConf.create({
            "model": {"lr": 0.1},
        })
        handler = DictConfigHandler(cfg=cfg)
        params = handler.build_dataclass(ModelParams, key="model")
        assert params.lr == 0.1
        assert params.batch_size == 32  # default
        assert params.name == "default"  # default

    def test_build_dataclass_nested_key(self):
        cfg = OmegaConf.create({
            "train": {
                "arch": {"depth": 18, "width": 128},
            },
        })
        handler = DictConfigHandler(cfg=cfg)
        arch = handler.build_dataclass(NestedConfig, key="train.arch")
        assert arch.depth == 18
        assert arch.width == 128

    def test_build_dataclass_missing_key_raises(self):
        cfg = OmegaConf.create({"other": 1})
        handler = DictConfigHandler(cfg=cfg)
        with pytest.raises(ValueError, match="Configuration key 'model' not found"):
            handler.build_dataclass(ModelParams, key="model")

    def test_build_dataclass_empty_allowed(self):
        """When key is missing, is_empty_allowed=True builds from empty config."""
        cfg = OmegaConf.create({})
        handler = DictConfigHandler(cfg=cfg)
        params = handler.build_dataclass(
            ModelParams, key="model", is_empty_allowed=True
        )
        assert params.lr == 0.001
        assert params.batch_size == 32
        assert params.name == "default"

    def test_build_dataclass_scalar_raises_type_error(self):
        cfg = OmegaConf.create({"model": 42})
        handler = DictConfigHandler(cfg=cfg)
        with pytest.raises(TypeError, match="must be a DictConfig"):
            handler.build_dataclass(ModelParams, key="model")


# --- get_listconfig_value tests ---
class TestGetListconfigValue:
    def test_get_listconfig_value_exists(self):
        cfg = OmegaConf.create({"layers": [64, 128, 256]})
        handler = DictConfigHandler(cfg=cfg)
        layers = handler.get_listconfig_value("layers")
        assert list(layers) == [64, 128, 256]

    def test_get_listconfig_value_missing_raises(self):
        cfg = OmegaConf.create({"foo": 1})
        handler = DictConfigHandler(cfg=cfg)
        with pytest.raises(ValueError, match="Configuration key 'layers' not found"):
            handler.get_listconfig_value("layers")

    def test_get_listconfig_value_wrong_type_raises(self):
        cfg = OmegaConf.create({"layers": {"depth": 18}})
        handler = DictConfigHandler(cfg=cfg)
        with pytest.raises(TypeError, match="must be a ListConfig"):
            handler.get_listconfig_value("layers")
