"""CTA-QSAR exception hierarchy."""


class CTAQSARError(Exception):
    """Base class for all CTA-QSAR errors."""


class ConfigurationError(CTAQSARError):
    """Invalid or missing configuration."""


class DataError(CTAQSARError):
    """Invalid dataset or data loading failure."""


class ChemistryError(CTAQSARError):
    """Chemical parsing/standardization failure."""


class EndpointDetectionError(CTAQSARError):
    """Endpoint could not be detected."""


class PluginError(CTAQSARError):
    """Plugin registration, discovery, or instantiation failure."""


class PluginUnavailableError(PluginError):
    """A plugin is registered but its optional dependencies are missing."""


class LLMError(CTAQSARError):
    """LLM provider call failed."""


class LLMOutputError(LLMError):
    """LLM returned malformed or unserializable output."""


class ExperimentError(CTAQSARError):
    """Experiment execution failure."""


class BudgetExhaustedError(CTAQSARError):
    """Compute or experiment budget exhausted."""


class ReproducibilityError(CTAQSARError):
    """Failed to record or verify provenance."""