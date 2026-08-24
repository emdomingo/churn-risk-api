"""The HTTP edge: validate, call the core, serialise. No churn logic lives here.

Every route is three lines because that is the architectural claim -- `scoring` and
`explain` are pure functions over an artifact, and this module is a shell over them. If a
route ever needs a fourth line of reasoning, that reasoning belongs in the core where it
can be tested without a web framework.

**The artifact loads at import, not per request.** In Lambda, module import happens in
the init phase, which is billed differently from invocation and, more importantly, only
happens on a cold start. Deferring the ~0.6 MB joblib load to the first request would put
it on a user's latency instead. `get_artifact` is cached so the dependency injection
below costs a dict lookup per request rather than a load.

**Why the import-time load is allowed to fail.** `artifacts/` is gitignored, so CI and
the test suite import this module with no artifact on disk. A hard failure at import
would make the module untestable without first training a model, which is exactly what
`tests/conftest.py` exists to avoid. So the warm-up is best-effort and a missing artifact
surfaces as a 503 at request time instead. In the deployed image the file is always
present -- the build's first stage trains it -- so the degraded path is a local and CI
convenience, never a production state.
"""

import logging
import time
import uuid
from functools import cache
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from churn.artifact import ChurnArtifact, load
from churn.config import Settings, get_settings
from churn.explain import explain
from churn.logging_setup import configure_logging
from churn.recommend import recommend
from churn.schemas import (
    DriverModel,
    ExplainResponse,
    HealthResponse,
    RecommendResponse,
    ScoredCustomer,
    ScoreRequest,
    ScoreResponse,
    SingleCustomerRequest,
)
from churn.scoring import score, score_one

logger = logging.getLogger("churn.api")

REQUEST_ID_HEADER = "x-request-id"


@cache
def get_artifact() -> ChurnArtifact:
    """The loaded model, once per process.

    A FastAPI dependency rather than a module global, so tests can substitute a fixture
    artifact through `app.dependency_overrides` without patching imports or touching
    disk. S5's Anthropic client will hang off the same mechanism.
    """
    return load(get_settings().artifact_path)


def artifact_dependency() -> ChurnArtifact:
    """Wrap the loader so a missing artifact is a 503, not a 500.

    A `FileNotFoundError` escaping to the framework would be an unhandled exception --
    which reads as a bug in the service. A model that has not been built is a service
    that is not ready, and that distinction is what an operator needs from the status
    code.
    """
    try:
        return get_artifact()
    except FileNotFoundError as exc:
        logger.error("artifact unavailable", extra={"error": type(exc).__name__})
        raise HTTPException(status_code=503, detail="Model artifact is not available.") from exc


@cache
def get_llm_client() -> anthropic.Anthropic | None:
    """The Anthropic client, or `None` when no key is configured.

    `None` is a supported state, not an error: `/score` and `/explain` do not need a key,
    and `/recommend` degrades rather than failing. Returning `None` here instead of
    raising is what makes "no key in the environment" behave identically to "the LLM is
    down" -- one degradation path, tested once.

    Cached for the same reason the artifact is: the client holds a connection pool, and
    rebuilding it per request would add a TLS handshake to every recommendation.
    """
    key = get_settings().anthropic_api_key
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


Artifact = Annotated[ChurnArtifact, Depends(artifact_dependency)]
LlmClient = Annotated["anthropic.Anthropic | None", Depends(get_llm_client)]
Config = Annotated[Settings, Depends(get_settings)]

configure_logging(get_settings().log_level)

app = FastAPI(
    title="Churn risk scoring",
    description=(
        "Calibrated churn probability, the SHAP drivers behind it, and a retention "
        "action drafted from a closed catalogue."
    ),
    version="0.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    """One structured line per request: id, route, status, latency.

    The request id is taken from the caller's header when present so a client can
    correlate its own logs with CloudWatch, and generated otherwise. It is echoed back on
    the response for the same reason.

    Nothing about the request *body* is logged here -- see `logging_setup`. The
    middleware only ever sees the route and the outcome, which is the cheapest way to
    guarantee that.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    started = time.perf_counter()

    response = await call_next(request)

    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(artifact: Artifact) -> HealthResponse:
    """Readiness, not liveness.

    It depends on the artifact deliberately: a health check that answers `ok` while the
    model is missing tells a load balancer to send traffic that can only fail. Reporting
    the version here is also the deploy verification step -- `model_version` changing is
    how you know the new image is the one serving.
    """
    return HealthResponse(model_version=artifact.model_version)


@app.post("/score", response_model=ScoreResponse)
def score_customers(request: ScoreRequest, artifact: Artifact) -> ScoreResponse:
    """List in, list out, in request order.

    One `transform` and one `predict_proba` for the whole batch -- `scoring.score` is
    vectorised, so this route does not loop over customers and neither should anything
    calling it.
    """
    records = [customer.to_record() for customer in request.customers]
    scores = score(artifact, records)

    logger.info(
        "scored",
        extra={"batch_size": len(scores), "model_version": artifact.model_version},
    )
    return ScoreResponse(
        model_version=artifact.model_version,
        results=[
            ScoredCustomer.from_score(result, customer.customerID)
            for result, customer in zip(scores, request.customers, strict=True)
        ],
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_customer(request: SingleCustomerRequest, artifact: Artifact) -> ExplainResponse:
    """Score plus the top drivers, for exactly one customer.

    `explain` rejects a batch rather than silently explaining row 0, which is why the
    schema takes a single customer instead of reusing `ScoreRequest`.

    The probability is calibrated; the driver contributions are in pre-calibration margin
    units. That asymmetry is real and documented in `explain` -- it is not a bug to be
    tidied up by rescaling the contributions.
    """
    record = request.customer.to_record()
    result = score_one(artifact, record)
    drivers = [DriverModel.from_driver(driver) for driver in explain(artifact, record)]

    logger.info(
        "explained",
        extra={
            "band": result.band.value,
            "probability": round(result.probability, 4),
            "model_version": artifact.model_version,
        },
    )
    return ExplainResponse(
        model_version=artifact.model_version,
        customerID=request.customer.customerID,
        probability=result.probability,
        band=result.band,
        drivers=drivers,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend_for_customer(
    request: SingleCustomerRequest, artifact: Artifact, llm: LlmClient
) -> RecommendResponse:
    """Score, drivers, and one retention action from the closed catalogue.

    **This route cannot 500 on an LLM failure.** `recommend` converts every upstream
    problem into an action-less result with a reason, so the score and drivers -- the
    parts that need no network -- are returned either way. The status code stays 200
    because the request was served: the caller got everything the service could compute.

    The LLM sees only the band, the three drivers, and the catalogue. It never receives
    the customer record or the id, and its choice is checked against the catalogue before
    it reaches this response.
    """
    record = request.customer.to_record()
    result = score_one(artifact, record)
    drivers = explain(artifact, record)

    recommendation = recommend(llm, result.band, drivers)

    logger.info(
        "recommended",
        extra={
            "band": result.band.value,
            "probability": round(result.probability, 4),
            "model_version": artifact.model_version,
            # The chosen action, or the reason there is none -- never the customer.
            "error": recommendation.reason,
        },
    )
    return RecommendResponse(
        model_version=artifact.model_version,
        customerID=request.customer.customerID,
        probability=result.probability,
        band=result.band,
        drivers=[DriverModel.from_driver(driver) for driver in drivers],
        action=recommendation.action,
        rationale=recommendation.rationale,
        drivers_used=list(recommendation.drivers_used),
        reason=recommendation.reason,
    )


def _warm() -> None:
    """Load the artifact during the Lambda init phase. See the module docstring."""
    try:
        get_artifact()
    except FileNotFoundError:
        logger.warning("starting without a model artifact; endpoints will return 503")


_warm()
