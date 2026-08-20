from cta_qsar.models.base import XGBoostPlugin


class XGBoostModel(XGBoostPlugin):
    """XGBoost plugin (name: ``xgboost``)."""


PLUGINS = [XGBoostModel]