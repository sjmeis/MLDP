from .mechanisms import (
    MultivariateCalibrated,
    TruncatedGumbel,
    VickreyMechanism,
    TEM,
    Mahalanobis,
    SynTF
)

__version__ = "0.1.0"

MECHANISMS = {
    "multivariate_calibrated": MultivariateCalibrated,
    "truncated_gumbel": TruncatedGumbel,
    "vickrey": VickreyMechanism,
    "tem": TEM,
    "mahalanobis": Mahalanobis,
    "syntf": SynTF
}

def get_mechanism(name: str, **kwargs):
    """
    Factory helper to quickly initialize an LDP mechanism by name.
    
    Example:
        >>> import mldp
        >>> engine = mldp.get_mechanism("vickrey", epsilon=1.5)
        >>> engine.replace_word("apple")
    """
    name_clean = name.lower().replace("-", "_").strip()
    if name_clean not in MECHANISMS:
        raise ValueError(f"Unknown mechanism '{name}'. Available: {list(MECHANISMS.keys())}")
    
    return MECHANISMS[name_clean](**kwargs)

# Expose everything cleanly at top-level package namespace
__all__ = [
    "MultivariateCalibrated",
    "TruncatedGumbel",
    "VickreyMechanism",
    "TEM",
    "Mahalanobis",
    "SynTF",
    "get_mechanism"
]