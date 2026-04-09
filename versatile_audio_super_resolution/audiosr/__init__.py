from .utils import seed_everything, save_wave, get_time, get_duration, read_list

def __getattr__(name):
    """Lazy import pipeline exports to avoid circular imports."""
    from . import pipeline as _pipeline
    val = getattr(_pipeline, name, None)
    if val is not None:
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
