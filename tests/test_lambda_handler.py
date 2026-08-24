"""The AWS entry point is wiring, so the tests are about wiring.

There is no Lambda here and no container: the image is verified by building it and
invoking through the Runtime Interface Emulator, which CI cannot do without Docker. What
CI *can* protect is the contract this module makes with the init phase -- that importing
it loads the artifact, and that no lifespan hook is left to run inside the first request.

Both import tests run in a subprocess. Import-time side effects happen once per
interpreter, and by the time pytest reaches this module `churn.api` is long since
imported and its cache warm, so an in-process assertion would only be reading the test
session's own history.
"""

import os
import subprocess
import sys
from pathlib import Path

import churn.api
from churn.artifact import ChurnArtifact, save
from churn.lambda_handler import handler

# Report what the init phase left behind: the artifact cache size, which is 1 only if
# module import actually loaded a model.
PROBE = (
    "import churn.lambda_handler as h, churn.api as a; "
    "print(a.get_artifact.cache_info().currsize)"
)


def _import_probe(artifact_path: Path) -> subprocess.CompletedProcess[str]:
    # The parent environment is inherited rather than replaced: on Windows an empty PATH
    # costs the interpreter its own DLLs. `ARTIFACT_PATH` is set on top, which outranks
    # any `.env` a developer happens to have.
    env = os.environ | {"ARTIFACT_PATH": str(artifact_path)}
    return subprocess.run(
        [sys.executable, "-c", PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_handler_wraps_the_application_with_lifespan_disabled() -> None:
    """Lambda has no server lifecycle, so ASGI lifespan must stay off.

    With it on, FastAPI would run startup inside the first *invocation* rather than the
    init phase, moving any warm-up onto a user's latency -- and register a shutdown hook
    the platform never calls, since containers are frozen and killed without notice.
    """
    assert handler.app is churn.api.app
    assert handler.lifespan == "off"


def test_importing_the_handler_loads_the_artifact(
    tiny_artifact: ChurnArtifact, tmp_path: Path
) -> None:
    """The init-phase claim, asserted rather than assumed.

    Lambda imports the handler during init and calls it during invocation. If the load
    ever moved to a request-time hook, the cache would still be empty at import and this
    would catch it -- which is the whole latency argument in one assertion.
    """
    path = save(tiny_artifact, tmp_path / "model.joblib")

    result = _import_probe(path)

    assert result.returncode == 0, result.stderr
    # The probe's own line is last: JSON log lines share stdout with it, which is the
    # point of the arrangement -- CloudWatch reads what the process prints.
    assert result.stdout.splitlines()[-1] == "1"


def test_importing_the_handler_survives_a_missing_artifact(tmp_path: Path) -> None:
    """A model that has not been built must not break import.

    CI imports this package with no artifact on disk, and so does every developer before
    their first training run. The degraded path is a 503 at request time, decided in
    `api.artifact_dependency` -- not an exception during init.
    """
    result = _import_probe(tmp_path / "absent.joblib")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "0"
    assert "endpoints will return 503" in result.stdout
