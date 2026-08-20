from cta_qsar.models.base import ElasticNetPlugin


class ElasticNetModel(ElasticNetPlugin):
    """ElasticNet regression plugin (name: ``elastic_net``)."""


PLUGINS = [ElasticNetModel]