from cta_qsar.models.base import LightGBMPlugin


class LightGBMModel(LightGBMPlugin):
    """LightGBM plugin (name: ``lightgbm``); registered only if installed."""


PLUGINS = [LightGBMModel]