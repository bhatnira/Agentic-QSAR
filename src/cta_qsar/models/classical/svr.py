from cta_qsar.models.base import SVRPlugin


class SVRModel(SVRPlugin):
    """SVR/SVC plugin (name: ``svr``)."""


PLUGINS = [SVRModel]