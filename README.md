# DictConfigHandler

## Overview

Reads values from an OmegaConf `DictConfig` by dot-notation key and builds dataclasses from config sections, so business logic does not depend on OmegaConf directly. Typed getters (`get_str_value`, `get_int_value`, `get_listconfig_value`, and so on) check the value's type, and missing keys raise `KeyError`.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/ry-yoshida-dev/DictConfigHandler.git
cd DictConfigHandler
pip install .
```

## Usage

```python
from dataclasses import dataclass

from omegaconf import OmegaConf

from dictconfig_handler import DictConfigHandler


@dataclass
class ModelParams:
    lr: float = 0.001
    batch_size: int = 32
    name: str = "default"


cfg = OmegaConf.create({"model": {"lr": 0.01, "name": "resnet"}, "layers": [64, 128]})
handler = DictConfigHandler(cfg=cfg)

params = handler.build_dataclass(ModelParams, key="model")
# ModelParams(lr=0.01, batch_size=32, name='resnet')

name = handler.get_str_value("model.name")  # "resnet"
layers = handler.get_listconfig_value("layers")  # [64, 128]
defaults = handler.build_dataclass(ModelParams, key="missing", is_empty_allowed=True)
# ModelParams(lr=0.001, batch_size=32, name='default')
```
