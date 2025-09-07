def smart_getattr(mod, candidates: Sequence[str]) -> Optional[Callable]:
    """Return first existing attribute (callable) by trying candidates in order."""
    for name in candidates:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    return None

def fail_with_attributes(mod, purpose: str) -> None:
    attrs = [a for a in dir(mod) if not a.startswith("_")]
    raise AttributeError(
        f"Cannot find a suitable function for {purpose} in module {mod.__name__}.\n"
        f"Available attributes:\n- " + "\n- ".join(attrs)
    )
