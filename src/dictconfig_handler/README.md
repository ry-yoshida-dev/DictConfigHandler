# dictconfig_handler

## Overview

Reads typed values and builds dataclasses from an OmegaConf `DictConfig`.

## Components

| Component | Description |
|-----------|-------------|
| [handler.py](./handler.py) | `DictConfigHandler`: looks up values by dot-notation key, checks their types, and builds dataclasses from sections. |
| [py.typed](./py.typed) | PEP 561 marker for type checkers. |
