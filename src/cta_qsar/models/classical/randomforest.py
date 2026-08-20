from cta_qsar.models.base import RandomForestPlugin


class RandomForestModel(RandomForestPlugin):
    """RandomForest plugin (name: ``random_forest``)."""


PLUGINS = [RandomForestModel]