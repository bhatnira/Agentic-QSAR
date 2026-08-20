from cta_qsar.models.base import RidgePlugin


class RidgeModel(RidgePlugin):
    """Ridge regression plugin (name: ``ridge``)."""


PLUGINS = [RidgeModel]