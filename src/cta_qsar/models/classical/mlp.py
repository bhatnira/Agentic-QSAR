from cta_qsar.models.base import MLPPlugin


class MLPModel(MLPPlugin):
    """Scikit-learn MLP plugin (name: ``mlp``); CPU-compatible deep feedforward net."""


PLUGINS = [MLPModel]