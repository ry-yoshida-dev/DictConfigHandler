from dataclasses import dataclass

import pytest
from omegaconf import DictConfig, OmegaConf

from dictconfig_handler import DictConfigHandler


@dataclass
class ModelParams:
    lr: float = 0.001
    batch_size: int = 32
    name: str = "default"


@dataclass
class NestedConfig:
    depth: int = 1
    width: int = 64


def make_handler(content: dict[str, object]) -> DictConfigHandler:
    return DictConfigHandler(cfg=OmegaConf.create(content))


class TestGetValue:
    def test_returns_sub_tree(self) -> None:
        handler = make_handler({"model": {"lr": 0.01, "batch_size": 64}})
        sub = handler.get_value("model")
        assert isinstance(sub, DictConfig)
        assert sub["lr"] == 0.01
        assert sub["batch_size"] == 64

    def test_resolves_nested_key(self) -> None:
        handler = make_handler({"a": {"b": {"c": 42}}})
        assert handler.get_value("a.b.c") == 42

    def test_missing_key_raises(self) -> None:
        handler = make_handler({"foo": 1})
        with pytest.raises(KeyError, match="Configuration key 'missing' not found"):
            _ = handler.get_value("missing")

    @pytest.mark.parametrize("value", [None, "???"])
    def test_null_and_missing_marker_are_not_found(self, value: object) -> None:
        handler = make_handler({"foo": value})
        with pytest.raises(KeyError):
            _ = handler.get_value("foo")

    def test_missing_key_with_empty_allowed_returns_empty_config(self) -> None:
        handler = make_handler({"foo": 1})
        sub = handler.get_value("missing", is_empty_allowed=True)
        assert isinstance(sub, DictConfig)
        assert len(sub) == 0


class TestTypedValue:
    def test_returns_each_type(self) -> None:
        handler = make_handler(
            {"s": "resnet", "i": 64, "f": 0.01, "b": True, "d": {"x": 1}, "l": [1, 2]}
        )
        assert handler.get_str_value("s") == "resnet"
        assert handler.get_int_value("i") == 64
        assert handler.get_float_value("f") == 0.01
        assert handler.get_bool_value("b") is True
        assert handler.get_dictconfig_value("d")["x"] == 1
        assert list(handler.get_listconfig_value("l")) == [1, 2]

    def test_missing_key_raises(self) -> None:
        handler = make_handler({"foo": 1})
        with pytest.raises(KeyError, match="Configuration key 'layers' not found"):
            _ = handler.get_listconfig_value("layers")

    def test_wrong_type_raises(self) -> None:
        handler = make_handler({"layers": {"depth": 18}})
        with pytest.raises(TypeError, match="must be ListConfig, got DictConfig"):
            _ = handler.get_listconfig_value("layers")

    def test_float_rejects_int(self) -> None:
        handler = make_handler({"lr": 1})
        with pytest.raises(TypeError, match="must be float, got int"):
            _ = handler.get_float_value("lr")


class TestBuildDataclass:
    def test_builds_from_key(self) -> None:
        handler = make_handler(
            {"model": {"lr": 0.01, "batch_size": 64, "name": "resnet"}}
        )
        params = handler.build_dataclass(ModelParams, key="model")
        assert params == ModelParams(lr=0.01, batch_size=64, name="resnet")

    def test_partial_section_uses_defaults(self) -> None:
        handler = make_handler({"model": {"lr": 0.1}})
        params = handler.build_dataclass(ModelParams, key="model")
        assert params == ModelParams(lr=0.1)

    def test_builds_from_nested_key(self) -> None:
        handler = make_handler({"train": {"arch": {"depth": 18, "width": 128}}})
        arch = handler.build_dataclass(NestedConfig, key="train.arch")
        assert arch == NestedConfig(depth=18, width=128)

    def test_missing_key_raises(self) -> None:
        handler = make_handler({"other": 1})
        with pytest.raises(KeyError, match="Configuration key 'model' not found"):
            _ = handler.build_dataclass(ModelParams, key="model")

    def test_missing_key_with_empty_allowed_uses_defaults(self) -> None:
        handler = make_handler({})
        params = handler.build_dataclass(
            ModelParams, key="model", is_empty_allowed=True
        )
        assert params == ModelParams()

    @pytest.mark.parametrize("is_empty_allowed", [False, True])
    def test_scalar_section_raises(self, is_empty_allowed: bool) -> None:
        handler = make_handler({"model": 42})
        with pytest.raises(TypeError, match="must be a DictConfig"):
            _ = handler.build_dataclass(
                ModelParams, key="model", is_empty_allowed=is_empty_allowed
            )
