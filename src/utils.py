#!/usr/bin/env python3

import toml

DEFAULT_CONFIG = "config.toml"


def load_cofig(path=None):
    if path is None:
        path = DEFAULT_CONFIG
    return toml.load(path)
