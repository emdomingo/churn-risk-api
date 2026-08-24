"""Runtime configuration, read once from the environment.

Deliberately small. `SPEC.md` §6 names three Lambda env vars -- `MODEL_VERSION`,
`LOG_LEVEL`, `ANTHROPIC_API_KEY` -- and only two of them are runtime settings.

**`MODEL_VERSION` is not here, on purpose.** It is consumed at *training* time by
`artifact.resolve_model_version`, which stamps it into the artifact; that stamp is what
every response reports. Reading the same variable again at request time would create a
second source of truth that could disagree with the model actually doing the scoring --
exactly the silent wrong-answer failure the single-artifact design exists to prevent. The
variable is still set on the Lambda because the image's build stage trains there; nothing
in the request path reads it.

`.env` is read for local development only and is gitignored. In Lambda there is no file,
just the process environment, which `BaseSettings` reads identically.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from churn.artifact import ARTIFACT_PATH


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # The process environment carries far more than our three variables; ignoring
        # the rest is required, not lenient.
        extra="ignore",
    )

    log_level: str = "INFO"

    # Unused until S5 wires `/recommend`. Declared now so the env contract lives in one
    # place and `.env.example` has something to point at. `None` is a valid state: the
    # service must still serve `/score` and `/explain` without a key.
    anthropic_api_key: str | None = None

    # A seam, not a knob. Tests override the artifact through FastAPI's dependency
    # system rather than this; it exists so a local run can point at an artifact built
    # somewhere other than the default `artifacts/`.
    artifact_path: Path = Field(default=ARTIFACT_PATH)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Parse the environment once per process.

    Cached because this runs in the Lambda init phase and every subsequent request on a
    warm container should reuse it. `lru_cache` also makes it overridable in tests via
    `get_settings.cache_clear()`.
    """
    return Settings()
