
class PrecisionConfig:
    def __init__(self, forced_precision: set[str]):
        self._forced_precision_fields = forced_precision

    def is_force_fp32_enabled(self, tag: str):
        if tag in self._forced_precision_fields:
            return True

        return False

def should_force_fp32(tag : str, precision_config: PrecisionConfig | None = None):
    if precision_config is None:
        return False
    # Precision config exists and needs to be checked
    return precision_config.is_force_fp32_enabled(tag)