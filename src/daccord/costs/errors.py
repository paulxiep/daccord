class CapExceeded(RuntimeError):
    """Raised when a recorded or estimated call would push today's per-provider
    spend over the configured daily cap. Bypass with env DACCORD_COSTS_OVERRIDE=1."""


class UnknownModel(KeyError):
    """Raised when a (provider, model) pair is not in costs/config.toml. Add the
    model to the [pricing.<provider>."<model>"] table — no silent zero-cost fallback."""
