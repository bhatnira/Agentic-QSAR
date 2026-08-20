from cta_qsar.models.base import ExtraTreesPlugin


class ExtraTreesModel(ExtraTreesPlugin):
    """ExtraTrees plugin (name: ``extra_trees``)."""


PLUGINS = [ExtraTreesModel]