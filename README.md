# DictConfigHandler

Utility for handling OmegaConf DictConfig. Retrieve sub-configurations by key and build dataclasses from config sections. Use it to isolate OmegaConf dependency from business logic.

## Requirements

- Python 3.10+
- omegaconf
- dataclass-initializer ([DataclassInitializer](https://github.com/ry-yoshida-private/DataclassInitializer))

## Setup

```bash
pip install -e .
```

For development (e.g. running tests):

```bash
pip install -e ".[dev]"
```

## Available methods

| Method | Description |
|--------|-------------|
| `get_value(key, is_empty_allowed=False)` | Retrieve any configuration value by dot-notation key. |
| `get_str_value(key)` / `get_int_value(key)` / `get_float_value(key)` / `get_bool_value(key)` | Retrieve a typed scalar at the given key. |
| `get_dictconfig_value(key)` | Retrieve a `DictConfig` sub-tree at the given key. |
| `get_listconfig_value(key)` | Retrieve a `ListConfig` sequence at the given key. |
| `build_dataclass(cls_, key, is_empty_allowed=False)` | Build a dataclass from the `DictConfig` section at the given key. |

## Example

From the project root or after installing the package:

```python
from dataclasses import dataclass
from omegaconf import OmegaConf

from dictconfig_handler import DictConfigHandler


@dataclass
class ModelParams:
    lr: float = 0.001
    batch_size: int = 32
    name: str = "default"


cfg = OmegaConf.create({
    "model": {
        "lr": 0.01,
        "batch_size": 64,
        "name": "resnet",
    },
})
handler = DictConfigHandler(cfg=cfg)

# Retrieve sub-configuration
sub = handler.get_value("model")
assert sub["lr"] == 0.01

# Build dataclass from config section
params = handler.build_dataclass(ModelParams, key="model")
assert params.lr == 0.01
assert params.batch_size == 64
assert params.name == "resnet"
```

- **key** – Dot-notation key path (e.g. `"model"`, `"train.arch"`).
- **is_empty_allowed** – If `True`, missing keys yield an empty DictConfig instead of an error; with `build_dataclass`, the instance is built from dataclass defaults only.
