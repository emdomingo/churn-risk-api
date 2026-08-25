# Project init + scaffold breakdown — portfolio-churn

## Context

`SPEC.md` describes a churn-risk microservice (score + SHAP explanation + LLM-drafted
retention action) deployed as a container-image Lambda behind API Gateway, with
GitHub Actions CI/CD. The repo is currently **empty** — `main` exists with zero
commits and only `SPEC.md` untracked.

**Status (S3 complete).** S0-S3 are done: `data.py`, `features.py`, `train.py`,
`artifact.py`, `scoring.py`, `explain.py`, and the notebook, with 77 tests green and
ruff clean. All five decision gates below are resolved. S4 (contracts + API) is next.
The "repo is currently empty" line above describes the state this plan was written in,
not the state today.

This plan covers what the SPEC deliberately leaves open: how to initialise the
project, and how to cut the work into scaffolds that each end at a runnable,
testable state. The SPEC's §11 guardrail ("surface tradeoffs, don't silently
choose") is honoured by listing the remaining decisions as explicit gates rather
than baking them in.

The organising principle: **pure core, thin edges.** All churn logic (data,
features, scoring, SHAP→drivers, action selection) lives in plain Python
functions with no web framework and no AWS. FastAPI, Mangum, and the Anthropic
client are thin shells over that core. This makes the test suite fast and
AWS-free, and makes the "explain the request path" gate easy to answer.

## Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Artifact flow | Train inside the Docker build (multi-stage) | Reproducible from repo alone; nothing committed; no extra AWS resources |
| IaC | AWS SAM (`template.yaml`) | One template for Lambda + HTTP API + IAM + logs; `sam local start-api` for local Lambda emulation |
| Local Docker | Install Docker Desktop before Night 2 | Required to build/test the image and by `sam build` for container images |
| Env/deps | `uv` + committed `uv.lock` | Already installed; lockfile satisfies the SPEC's "pin all versions" |
| Python | 3.11 | Matches local 3.11.11. **Revised (S6):** the AWS Lambda 3.11 base image proved unusable (AL2 glibc 2.26 vs xgboost's `manylinux_2_28` wheels); the Python version stands, the base image does not |
| Training data | Commit the full CSV (~1 MB) | Hermetic build, no network on the deploy path, exact training bytes in the repo. **Deliberate exception to §2/§11** — see below |

**On committing the CSV.** §2 and §11 say don't commit the full dataset. That
guardrail exists to keep multi-hundred-MB data and model artifacts out of git;
at ~1 MB the reasoning doesn't reach. Committing it removes
`raw.githubusercontent.com` from the deploy path — every merge to `main` rebuilds
the image, so a fetch step would put an external host between you and a
successful deploy, and §10 says deploy is the one thing that must never break.
It also makes reproducibility stronger, not weaker: the exact bytes trained on
are in the repo. The guardrail still holds for the model artifact, which is where
it actually bites. Note this exception explicitly in the README so it reads as a
decision rather than an oversight. `SPEC.md` §2 and §11 now diverge from this
plan — amend both lines as part of S0 so the spec stays the source of truth.

**Prerequisites to arrange before Night 3:** an AWS account, Docker Desktop
(WSL2 backend), AWS CLI v2, SAM CLI, and a GitHub remote for this repo. None are
installed today.

## Repo layout

```
pyproject.toml            uv project; dep groups: default(runtime) / train / dev
uv.lock                   committed — reproducibility
src/churn/
  __init__.py
  config.py               env settings (MODEL_VERSION, LOG_LEVEL, ANTHROPIC_API_KEY)
  logging_setup.py        ~20-line JSON formatter, no new dep
  data.py                 load CSV, TotalCharges handling, train/val/test split
  features.py             sklearn ColumnTransformer (fit at train, pickled)
  train.py                train → calibrate → fit SHAP → persist → metrics report
  artifact.py             load/save the single model artifact; MODEL_VERSION stamp
  scoring.py              predict_proba → score + risk band
  explain.py              TreeExplainer → top-N drivers (feature, value, contribution, direction)
  actions.py              closed action catalogue + validation
  recommend.py            prompt build → Anthropic call → parse → graceful degradation
  schemas.py              Pydantic request/response contracts
  api.py                  FastAPI app + 4 routes
  lambda_handler.py       Mangum(app, lifespan="off"); artifact loaded at import
tests/
  conftest.py             session fixture: trains a tiny model on the 200-row sample
  fixtures/telco_sample.csv   ~200 rows, committed
  test_data.py test_scoring.py test_explain.py test_api.py test_recommend.py
data/Telco-Customer-Churn.csv   committed, ~1 MB — the build's data input
infra/bootstrap.yaml      one-time: GitHub OIDC provider, deploy role, ECR repo
template.yaml             SAM: Lambda (Image) + HTTP API + log group
Dockerfile                multi-stage: trainer → lambda runtime
.github/workflows/ci.yml  ruff + pytest, no AWS
.github/workflows/deploy.yml  OIDC → build → ECR → sam deploy
.env.example  .gitignore  README.md
```

`artifacts/` is gitignored — the model artifact never enters git. `data/` and the
200-row fixture are committed.

## Initialisation (Scaffold 0)

```powershell
uv init --python 3.11 --package --name churn      # creates pyproject + src/churn
uv add fastapi mangum pydantic pydantic-settings numpy pandas scikit-learn xgboost shap anthropic
uv add --group train matplotlib                    # reliability diagram only
uv add --group dev pytest pytest-cov ruff httpx
```

**PowerShell notes.** Commands in this plan are single-line on purpose — `\` is a
bash continuation and PowerShell passes it through as a literal argument (uv then
reads it as the path `C:\` and fails). PowerShell's continuation character is a
backtick. Also, `curl` in PowerShell 5.1 is an alias for `Invoke-WebRequest`, which
does *not* take `-X`/`-d` — use `curl.exe` explicitly for every `curl` in the
verification steps below, or run them from Git Bash.

Runtime group is deliberately fat: `sklearn` (unpickles the calibrator +
ColumnTransformer), `xgboost` (booster), `shap` (per-request `/explain`), and
`pandas` (the ColumnTransformer is fitted on named columns) are all genuinely
needed at inference. Only `matplotlib` is train-only. Expect a ~700MB–1.2GB
image — well under Lambda's 10GB, but it is the main cold-start driver; note it
in the README and size the function at 1536MB memory.

`.gitignore`: `artifacts/ .env .venv/ __pycache__/ *.egg-info .aws-sam/`
Then make the first commit (repo has no commits yet — no `git init` needed).

## Scaffold breakdown

Each scaffold ends green — runnable and tested — before the next starts. Hand
them to Claude Code one at a time.

### Night 1 — model + explainability

**S1 · Data layer** — `data.py`, `tests/test_data.py`, plus two committed files:
the full CSV in `data/` and the ~200-row fixture in `tests/fixtures/`.
Download the CSV once from
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
and commit it; no fetch code, no network at build or test time. `data.py` loads
from a path, coerces `TotalCharges`, handles the blank strings, drops
`customerID`, and does the stratified split on a fixed seed.
*Done when:* `pytest tests/test_data.py` passes and the loader returns a typed
frame with zero silent NaNs.

**S2 · Train + calibrate + SHAP** — `features.py`, `train.py`, `artifact.py`.
Stratified train/val/test with a fixed seed; XGBoost with `scale_pos_weight`;
`CalibratedClassifierCV(cv="prefit")` on the val split; `TreeExplainer` fitted
and persisted alongside. Single artifact file, version-stamped. Prints the
metrics table (ROC-AUC, PR-AUC, Brier, confusion matrix) and writes a reliability
diagram PNG for the README.
*Done when:* `uv run --group train python -m churn.train` produces
`artifacts/model.joblib` and a metrics block you can paste into the README.

**S3 · Inference core** — `scoring.py`, `explain.py`, `tests/conftest.py`,
`test_scoring.py`, `test_explain.py`.
Pure functions: record → score + band; record → top-N drivers. `conftest.py`
trains a tiny model on the 200-row fixture at session scope, so the whole suite
runs in seconds with **no committed artifact and no network** — this is what keeps
`ci.yml` AWS-free.
*Done when:* scores are in [0,1], band boundaries are tested at their edges, and
drivers come back with correct signs on a known-risky record.
**Done (S3).** All three met, 38 tests across `test_scoring.py` / `test_explain.py`.
Two shapes worth knowing before S4 wraps them in Pydantic: `score` is batch-first
(`score_one` is a one-line convenience over it) and `explain` *rejects* a batch rather
than silently explaining row 0. `select_drivers` takes a Series rather than a record so
the selection policy is testable with no model involved; it ranks by magnitude, keeps
the sign, and applies a noise floor **relative to each customer's own strongest driver**
(5%), returning fewer than `top_n` rather than padding the list with a contribution
`/recommend` would then have to justify. On the current test split that floor never
fires — it is a guard, not a filter.

### Night 2 — service + agent + container

**S4 · Contracts + API** — `schemas.py`, `config.py`, `logging_setup.py`,
`api.py`, `test_api.py`.
Strict Pydantic models; batch list-in/list-out on `/score`; `model_version` in
every response. `/health`, `/score`, `/explain`.
*Done when:* `uv run uvicorn churn.api:app --reload` serves all three and the
contract tests pass against the fixture.

**S5 · `/recommend`** — `actions.py`, `recommend.py`, `test_recommend.py`.
Closed catalogue (fee waiver, plan downgrade, contract-term incentive, support
callback, loyalty perk). Prompt receives *only* risk band + top-3 drivers +
catalogue. Validate the returned action against the catalogue — reject and
degrade if it's off-list. On any LLM failure: return score + drivers,
`action: null`, plus a reason; never 500.
*Done when:* tests with a mocked client prove (a) only catalogue actions escape,
(b) timeout/error/off-list responses all degrade to a 200.

**S6 · Container** — `Dockerfile`, `lambda_handler.py`, `.env.example`.
Stage 1 copies `data/` and trains; stage 2 is the Lambda base image and copies
only artifacts + runtime deps. No network needed during the build. Artifact loads
at module import so it lands in the Lambda init phase.
*Done when:* `docker build .` succeeds and `docker run -p 9000:8080` + the Lambda
RIE `curl` returns a real score.

### Night 3 — deploy

**S7 · Bootstrap infra** — `infra/bootstrap.yaml`, deployed once by hand.
Creates the GitHub OIDC identity provider, a deploy role with trust scoped to
this repo *and* `ref:refs/heads/main`, and the ECR repo with a lifecycle policy.
This must exist before `deploy.yml` can ever run — it is the chicken-and-egg
step people trip on.
*Done when:* `aws sts assume-role-with-web-identity` works from a scratch Actions
run, and the ECR repo is listed.

**S8 · SAM template + first manual deploy** — `template.yaml`.
`AWS::Serverless::Function` with `PackageType: Image`, 1536MB, ~30s timeout, env
vars (`MODEL_VERSION`, `LOG_LEVEL`, `ANTHROPIC_API_KEY`), an HTTP API event, and
an explicit log group with retention.
*Done when:* `sam deploy --guided` yields a live URL that answers `/health`.

**S9 · Pipelines** — `ci.yml` (ruff + pytest on PR, no AWS) and `deploy.yml`
(on push to `main`, needs CI green: OIDC → build → push to ECR → `sam deploy`).
*Done when:* a PR runs CI only; a merge deploys and the live URL serves the new
`model_version`.

**S10 · README + smoke test** — architecture diagram, `curl` for all four
routes, metrics table with one sentence each, the calibration and imbalance
decisions, the committed-CSV exception to §11, "what I'd do next". Plus one smoke
test hittable against the live URL.

## Decision gates — surface these, don't let them be chosen silently

1. **`TotalCharges` blanks** (~11 rows, all `tenure == 0`). Impute 0, impute
   `MonthlyCharges`, or drop? These are brand-new customers, so the choice is
   substantive. → S1.
   **Resolved (S1):** filled with **0.0** — the semantically correct amount for a
   customer billed for zero months, not an estimate. Keeps `tenure == 0` customers
   scoreable, which matters because they are a real retention target at inference.
2. **SHAP explains the *uncalibrated* margin.** `CalibratedClassifierCV` wraps
   the booster, and `TreeExplainer` runs on the booster underneath. So the
   returned *score* is calibrated while the *drivers* explain the pre-calibration
   margin. This is defensible (calibration is a monotone map, so the ranking of
   drivers is unchanged) but it must be stated in the README, not glossed. → S2/S3.
   **Resolved (S2/S3):** accepted as designed and documented at the top of both
   `train.py` and `explain.py`. Contributions are reported in **margin units**, so they
   do not sum to the probability; `test_contributions_reconstruct_the_margin` pins the
   additivity against the booster. Still owed a paragraph in the README.
3. **Driver naming after one-hot encoding.** Report level-specific names
   (`Contract=Month-to-month`) or aggregate contributions back to the original
   column (`Contract`)? Level-specific is more actionable for `/recommend`. → S3.
   **Resolved (S3): level-specific.** `/explain` returns `Contract=Month-to-month`;
   aggregating to `Contract` would leave `/recommend` unable to tell whether a
   contract-term incentive is the right offer or a pointless one. The name is
   reconstructed from the fitted `OneHotEncoder.categories_`, not by splitting the
   encoded name, so a column containing the separator cannot corrupt it.
4. **Risk-band thresholds.** Proposed starting point low <0.30 / med / high >0.60,
   but pick from the PR curve once S2's metrics exist. → S3.
   **Resolved (S3): low < 0.167 / medium / high >= 0.50**, not the proposed 0.30/0.60.
   The low cut *is* the cost-derived operating threshold from gate #5, so "band is not
   low" and "the model says intervene" are the same predicate and the confusion matrix
   in the metrics table describes exactly the medium+high population;
   `test_low_band_cut_is_the_cost_derived_operating_threshold` enforces that. The high
   cut comes off the test-split precision curve: at 0.50 the high band is 23% of
   customers and 63% of them churn, against a 26.5% base rate.
5. **Confusion-matrix threshold** — a business call (recall on churners vs.
   retention-offer cost), not a default. → S2.
   **Resolved (S2): derived, not picked.** `MISSED_CHURNER_COST_RATIO = 5.0` (missing a
   churner costs 5x a wasted retention offer) gives `1 / (1 + 5)` = **0.167** for a
   calibrated score. The number to argue with is the cost ratio, not the threshold.

### Raised and resolved during S6

6. **Lambda base image.** `public.ecr.aws/lambda/python:3.11` is Amazon Linux 2, glibc
   2.26. xgboost publishes linux wheels as `manylinux_2_28` only — 3.2.0 and every later
   release — so there is no installable wheel: the build falls back to the sdist and dies
   on a missing cmake. numpy, scipy, sklearn and pandas are unaffected; they still ship
   `manylinux_2_17`. Three ways out, each with a cost.
   **Resolved (S6): `python:3.11-slim-bookworm` + `awslambdaric`.** Debian slim is glibc
   2.36, so every locked wheel installs and xgboost 3.2.0 — the version the recorded
   metrics were measured on — stays put. The AL2023-based 3.13 image would also have
   worked, but drags xgboost to 3.4.1 and numpy to 2.5.2, invalidating those numbers;
   compiling xgboost adds a C++ toolchain and ~10 minutes to every lock change. The cost
   paid instead is wiring the runtime interface client by hand, which is four lines and
   an AWS-documented path.

7. **Image size.** The first working image came out at 2.1GB against the 700MB–1.2GB
   budget, and cold start is this design's known weak point. 454MB of it is
   `nvidia-nccl-cu12`, which xgboost declares on linux for distributed GPU training.
   **Resolved (S6): filtered out of the runtime install.** CPU inference never dlopen's
   those libraries — verified by deleting them from a built image and re-invoking, which
   returned byte-identical `/score` and `/explain` responses. The exported requirements
   are filtered before install, and `--no-deps` stops the resolver adding them back;
   `uv.lock` is untouched, so the image is a deliberate *subset* of the resolution rather
   than a different one. Result: **1.28GB**, and 326MB less downloaded per build.

### Raised and resolved during S8

8. **Lambda memory and timeout.** PLAN pencilled 1536MB and ~30s before anything ran.
   **Resolved (S8): 3008MB and 29s.** Memory is not headroom — measured peak is 341MB of
   whatever is allocated. It buys vCPU and IO bandwidth for the cold start, and duration
   billing means faster-and-bigger costs about the same as slower-and-smaller. 1769MB (one
   vCPU) was deployed first and measured; see gate 10. The timeout is 29s because that is
   API Gateway's HTTP API integration ceiling — anything above it is unreachable through
   the API, so matching makes the two limits one number. `recommend.py`'s comments were
   realigned from 30s to 29s.

9. **Log retention and public exposure.**
   **Resolved (S8): 14 days, and stage throttling at 5 rps / burst 10.** CloudWatch
   defaults to Never Expire, which is how a $0 project grows a bill. The URL is public and
   unauthenticated as SPEC intends — a demo anyone can curl is the point — but `/recommend`
   spends Anthropic credits per call, so every route is rate limited at the stage.
   Throttled requests are refused by API Gateway before Lambda is invoked, costing neither
   function time nor tokens. This bounds abuse rather than preventing it, which is the
   right trade for a portfolio endpoint.

10. **The cold start does not fit the init phase.** Measured on the deployed function:

    ```
    INIT_REPORT  Init Duration: 10006.25 ms  Status: timeout   (at 1769MB)
    INIT_REPORT  Init Duration:  9997.64 ms  Status: timeout   (at 3008MB)
    REPORT       Duration: 27098 ms  Memory 1769 MB  Max Used 341 MB
    REPORT       Duration: 20487 ms  Memory 3008 MB  Max Used 330 MB
    ```

    Lambda caps the init phase at 10s. Over that, it *suppresses* init and re-runs the
    whole thing inside the invocation, so the wasted 10s is added to the real work: 10 +
    20.5 = 30.5s, past API Gateway's 29s. **The first request after a deploy therefore
    returns 503; every request after it is 70–160ms.**

    Where the ~20s goes — measured inside the image, local CPU:

    | step | local |
    |---|---|
    | `import pandas` | 1.06s |
    | `import sklearn` | 1.40s |
    | `import shap` (pulls numba/llvmlite) | 0.47s |
    | `import anthropic` | 0.56s |
    | `import fastapi` + `mangum` | 0.23s |
    | `import xgboost` | 0.03s |
    | `joblib.load(model.joblib)` | **0.08s** |
    | total | 3.83s (≈20s on Lambda) |

    3.83s of local CPU becoming ~20s on Lambda means the cost is first-touch IO against a
    lazily streamed 1.49GB image, not compute — which is also why doubling memory bought
    only 6.6s. **The artifact load is irrelevant at 0.08s**, so "lazy-load the model" would
    buy nothing.

    Three levers were measured before accepting the behaviour:

    - **Precompiled bytecode — applied.** A Lambda filesystem is read-only, so Python can
      never cache bytecode at runtime; the image shipped without `.pyc` and recompiled the
      whole dependency tree on every cold start. 8.53s uncached against 2.40s cached;
      fresh-container init 6.0s → 4.1s. Costs ~150MB. This is in the Dockerfile.
    - **Tiered laziness (defer SHAP only) — rejected on measurement.** The score path
      without `shap` is 2.84s of the 3.67s total, so deferring SHAP leaves ~78% of the
      cost, ~15.5s on Lambda, still over the cap. It is also blocked by design: the
      artifact pickle *contains* a `TreeExplainer`, so `joblib.load` imports `shap`
      transitively — deleting `numba` from a built image breaks artifact loading outright.
      Splitting the artifact would undo the one-file no-drift property.
    - **Further image trimming — rejected on measurement.** Stripping every `.so` saves
      75MB of 853MB; dropping `.py` files that duplicate a `.pyc` would add 111MB but
      breaks tracebacks and any library that reads its own source. Best case ~1.3GB. The
      bulk is irreducible binary: xgboost 228MB, llvmlite 173MB, scipy 113MB. S6 had
      already taken the easy 454MB by filtering `nvidia-`.

    **Resolved (S8): accepted and documented, not worked around.** The remaining fix would
    be to defer the heavy imports out of module scope so the init phase stays trivial and
    the ~20s lands inside the 29s invocation budget — which converts the 503 into a slow
    200. That inverts the "artifact loads at import so it lands in the init phase"
    decision in `SPEC.md` §6, and the owner chose to document the behaviour rather than
    restructure the edge for it. A warmer (EventBridge pinging `/health`) and provisioned
    concurrency (~$19–32/month) were both rejected — the first papers over the cause, the
    second ends the free-tier story.

11. **Does CI build the container image?** Never building it means `main` breaks at deploy
    time, which is the one failure that costs a live URL. Building it on every run taxes a
    README-only pull request with a multi-minute build for no signal. → S9.
    **Resolved (S9): path-filtered.** The `changes` job in `ci.yml` diffs against the base
    and builds only when `Dockerfile`, `uv.lock`, `pyproject.toml`, or `.dockerignore`
    moved — those four are the entire break surface (the `nvidia-` filter, `--no-deps`, the
    Debian base-image choice). An unknown base (`workflow_dispatch`, a first push) builds
    rather than skips: an explicit manual run is a request for the expensive path, and a
    missing base is not evidence of no change. The build carries `--provenance=false
    --sbom=false` even though a local build does not need them, so a green CI build is
    evidence about the artifact that actually ships, and follows with an import check —
    a build that succeeds but produces an image that cannot import is the hole the build
    alone leaves open.

12. **Does `deploy.yml` pass `ANTHROPIC_API_KEY` on every deploy?** The function currently
    runs without a key and `/recommend` degrades to `action: null`, so the live demo
    undersells S5. Against that, the key becomes a Lambda env var. → S9.
    **Resolved (S9): passed, from a GitHub secret.** The exposure was checked rather than
    assumed. `AnthropicApiKey` is already `NoEcho: true` with a `""` default, so
    `DescribeStacks` returns `****`; the real plaintext surface is the Lambda env var via
    `lambda:GetFunctionConfiguration`. In this account only two principals hold that — the
    owner's admin identity, and `churn-github-deploy`, assumable only by STS on the exact
    subject `repo:emdomingo/churn-risk-api:ref:refs/heads/main`. Secrets Manager would
    close it properly but adds a resource, an IAM statement, and a cold-start fetch to a
    function whose cold start is already gate 10 — `SPEC.md` §1 scope drift for a
    rotatable key. The `""` default is kept, so a keyless deploy stays a supported state
    rather than a pipeline failure. Corollary, deliberate: because the parameter is passed
    every time, deploying with the secret removed *clears* the live key.

## Verification

- **Per scaffold:** `uv run pytest` green, `uv run ruff check` clean.
- **Local service:** `uv run uvicorn churn.api:app` then `curl` each of the four
  routes with a fixture record; confirm `model_version` in every payload.
- **Container:** `docker build .` then `docker run -p 9000:8080` against the
  Lambda Runtime Interface Emulator; same four `curl`s through the RIE event shape.
- **LLM degradation:** run `/recommend` locally with `ANTHROPIC_API_KEY` unset —
  must return 200 with `action: null` and a reason.
- **Deployed:** `curl <live-url>/health` returns the expected `model_version`;
  then the same four routes against the live URL; confirm structured JSON lines
  in CloudWatch carry request id, latency, and score — and no raw PII.
- **Pipeline:** open a PR (CI runs, no AWS creds touched), merge (deploy runs,
  live `model_version` changes).
