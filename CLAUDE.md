# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A churn-risk scoring microservice: given a customer record it returns a calibrated
probability, the SHAP drivers behind it, and an LLM-drafted retention action chosen
from a closed catalogue. Deployed as a container-image Lambda behind API Gateway.

It is a **portfolio project on a 2–3 night budget**. Deployment and explainability are
the deliverable; the model is deliberately simple. `SPEC.md` §1 lists explicit
non-goals — SOTA accuracy, hyperparameter grids, a frontend, ensembles, streaming.
Resist drift toward any of them.

## Governing documents

Read both before making changes:

- **`SPEC.md`** — the requirements. Sections marked *"I must be able to explain X"* are
  gates, not aspirations.
- **`PLAN.md`** — the approved init and scaffold breakdown (S0–S10 across three nights),
  the locked decisions and why, and five open decision gates.

When a decision in `SPEC.md` is superseded, amend `SPEC.md` in place with an
`**Amended (Sn):**` marker rather than letting the spec silently diverge from the code.
Two such amendments already exist in §2 and §11.

## Working mode — the defining constraint

The repo owner writes none of the code but owns all of the understanding. Sections of
`SPEC.md` marked *"I must be able to explain X"* are gates on accepting work, not
aspirations. The practical consequence is the guardrails below — above all, surfacing a
tradeoff instead of resolving it yourself, which inverts the usual default of making
routine judgment calls on the user's behalf.

## Guardrails (`SPEC.md` §11)

- **Stop and surface tradeoffs** rather than silently choosing — impute vs. drop, SAM vs.
  Terraform, any threshold value. The owner owns those calls. All five gates `PLAN.md`
  opened are now resolved and marked there with `**Resolved (Sn):**`; S4-S10 will raise
  new ones (batch size limits, log redaction, LLM timeout and retry, Lambda memory), and
  they get the same treatment — surfaced with the evidence, decided by the owner.
- **Do not add dependencies** not needed for a listed feature.
- **Do not upgrade the model** beyond a single gradient-boosted tree — no ensembles, no
  stacking, no hyperparameter grids.
- **Do not commit** the model artifact (>a few MB), any secret, or `.env`. The ~1 MB
  training CSV *is* committed; that is a deliberate amendment, not a precedent.
- **Pin all versions.** `uv.lock` is committed; reproducibility is part of the grade.

## Commands

```powershell
uv sync --all-groups                       # install everything
uv run pytest                              # full suite
uv run pytest tests/test_data.py::test_x   # single test
uv run ruff check .                        # lint
uv run --group train python -m churn.train # train, calibrate, fit SHAP, write artifacts/
uv run uvicorn churn.api:app --reload      # serve locally
docker build -t churn:local .              # trains in stage 1; ~1.28GB image
```

The container is exercised through the Lambda Runtime Interface Emulator, mounted from
the host rather than baked into the image — the Dockerfile's closing comment carries the
`docker run` line. Note that `docker.exe` is not on PATH in this environment; it lives in
`$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin`.

Dependency groups: default = Lambda runtime, `train` = training-only (`matplotlib`),
`dev` = tooling. Adding a runtime dep grows the Lambda image, so put it in a group
deliberately.

## Windows / PowerShell

- `\` is **not** a line continuation in PowerShell — it gets passed through as a literal
  argument (`uv` then reads it as the path `C:\` and fails). Keep commands single-line.
  The continuation character is a backtick.
- `curl` in PowerShell 5.1 aliases `Invoke-WebRequest`, which does not accept `-X` or
  `-d`. Use `curl.exe` explicitly, or run from Git Bash.

## Architecture

**Pure core, thin edges.** All churn logic — data, features, scoring, SHAP→drivers,
action selection — lives in plain functions under `src/churn/` with no web framework and
no AWS. `api.py` (FastAPI), `lambda_handler.py` (Mangum), and the Anthropic client are
thin shells over that core. This is what keeps the test suite fast and AWS-free, and it
is the structure behind the "explain the request path" gate.

Request path: client → API Gateway (HTTP API) → Lambda cold start → Mangum → FastAPI →
core. The artifact loads at **module import**, not per-request, so it lands in the Lambda
init phase.

`PLAN.md` has the full file tree and the per-scaffold done-when criteria.

## Non-obvious constraints

- **SHAP explains the uncalibrated margin.** `CalibratedClassifierCV` wraps the booster;
  `TreeExplainer` runs on the booster underneath. So the returned *score* is calibrated
  while the *drivers* explain the pre-calibration margin. Defensible — calibration is
  monotone, so driver ranking is unaffected — but it must be stated in the README, never
  glossed.
- **No model artifact in git.** `artifacts/` is gitignored. Tests therefore never load a
  prebuilt model: `tests/conftest.py` trains a tiny one on the committed 200-row fixture
  at session scope. This is what lets `ci.yml` run with no AWS and no network.
- **The training CSV *is* committed** (~1 MB in `data/`), a deliberate exception to
  `SPEC.md` §11. It makes the Docker build hermetic and keeps an external host off the
  deploy path. Do not "fix" this by adding a fetch step.
- **The image is large** (1.28GB) because `shap`, `xgboost`, `sklearn`, and `pandas` are
  all genuinely needed at inference. That is the main cold-start driver; the function is
  sized at **3008MB** partly to compensate — Lambda scales vCPU with memory, and 1769MB
  was tried first and still overran the init cap (gate 8). It would be 2.1GB but for the
  `nvidia-` filter in the Dockerfile — see the S6 gates in `PLAN.md`.
- **The base image is `python:3.11-slim-bookworm`, not the AWS Lambda base.** The AWS
  Python 3.11 image is Amazon Linux 2 (glibc 2.26) and xgboost ships linux wheels as
  `manylinux_2_28` only, so nothing installs there without compiling. `awslambdaric` is
  wired in by hand as a consequence, and is a runtime dependency scoped to
  `sys_platform == 'linux'` so local Windows syncs skip it.
- **`/recommend` must never 500.** If the LLM call fails, return score + drivers with
  `action: null` and a reason. The LLM sees only risk band, top-3 drivers, and the action
  catalogue — it selects and justifies, it never invents actions or customer facts.
  Validate its choice against the catalogue before returning it.

## Current state

**S0–S10 complete.** 142 offline tests green plus 4 network smoke tests that skip without
`CHURN_SMOKE_URL`; ruff clean. The service is live at
`https://03bmm97sm6.execute-api.ap-southeast-2.amazonaws.com` in account 033177020842,
serving all four routes with `model_version` `0.1.0+3f5fb64`.

**S9 is proven end to end**: PR #1 ran CI only, the merge built, pushed, and deployed, and
`/recommend` returns a real catalogue action — confirming `ANTHROPIC_API_KEY` reaches the
function. The first merge failed at the OIDC step; see gate 13.

**S10 is written but not yet pushed.** `README.md` and `tests/test_smoke.py` are committed
locally and `main` is ahead of `origin`. Pushing triggers a deploy and moves
`model_version`.

Built so far: `src/churn/data.py` (`load_raw` / `clean` / `split`), the committed
`data/Telco-Customer-Churn.csv`, a 200-row fixture at `tests/fixtures/telco_sample.csv`
regenerable via `scripts/make_fixture.py`, `exploration.ipynb` (the pre-registered
univariate prior S2's SHAP output is checked against), the S2 model path — `features.py`
(one `ColumnTransformer`), `train.py` (train → calibrate → SHAP → persist → report),
`artifact.py` (single joblib, version-stamped) — the S3 inference core (`scoring.py`,
`explain.py`), the S4 edge (`schemas.py`, `config.py`, `logging_setup.py`, `api.py`), the
S5 agent (`actions.py`, `recommend.py`), and the S6 container (`Dockerfile`,
`lambda_handler.py`, `.env.example`, `.dockerignore`).

**Seven decision gates are resolved**; `PLAN.md` carries each with its reasoning — the
original five from S1–S3, plus the S6 pair (base image, image size). `TotalCharges`
blanks filled with 0.0 and `SeniorCitizen` normalised in `clean` (S1); confusion-matrix
threshold derived from `MISSED_CHURNER_COST_RATIO = 5.0` rather than picked (S2);
level-specific driver names and the risk bands below (S3); `MAX_BATCH_SIZE = 1000`
measured rather than guessed (S4); LLM model, timeout and retries set in `recommend.py`
against the 30s function budget (S5).

Current test metrics (seed 42): ROC-AUC 0.833, PR-AUC 0.638, Brier 0.141 (uncalibrated
0.168), against a tenure-only floor of 0.734. The SHAP top-10 matches the notebook's
prior and contains none of `gender` / `PhoneService` / `MultipleLines`. The in-image
training run reproduces the same top-10.

S2 decisions worth knowing before touching `train.py`:

- **Sigmoid, not isotonic** — ~1,400 val rows is too few for a staircase, and
  `scale_pos_weight` distortion is close to a constant log-odds shift, which is exactly
  what two parameters undo.
- **The tree count and depth come from a 20% stopping slice inside `train`, never `val`.**
  A fixed 300-tree budget was tried and measurably overfit. `val` is touched by nothing
  but the calibrator, which is what keeps the calibration story clean.
- **Two passes on purpose.** Early stopping leaves unused trees in the model and
  `TreeExplainer` would read them, so the shipped booster is refit at the chosen count.
  `test_explainer_explains_the_shipped_model` asserts SHAP additivity against the margin.

S3 decisions worth knowing before touching `scoring.py` / `explain.py`:

- **Risk bands are low < 0.167 / medium / high >= 0.50.** The low cut *is*
  `operating_threshold()`, so "not low" and "the model says intervene" are one predicate
  and the confusion matrix describes exactly the medium+high population.
  `test_low_band_cut_is_the_cost_derived_operating_threshold` fails if the two drift
  apart. `scoring.py` declares the constant itself rather than importing `train` — that
  module is offline code the Lambda image has no reason to load.
- **Drivers are named at the level** (`Contract=Month-to-month`), reconstructed from the
  fitted `OneHotEncoder.categories_` rather than by splitting the encoded name. Because
  `drop=None` keeps every level, a customer carries a contribution for levels they do
  *not* have — `Contract=Two year` as a protective driver means the model is pricing its
  absence. `Driver.value` reports the customer's raw value on the source column, so that
  pairing can read as `feature=Contract=Two year, value=Month-to-month`. Intended.
- **`select_drivers` ranks by magnitude, not by positive contribution.** For a low-risk
  customer the honest drivers are the ones keeping them; filtering to positives would
  manufacture a risk story out of near-zero noise and hand it to `/recommend`. It takes a
  Series, not a record, so the policy is testable with no model in the test.
- **The noise floor is relative** (5% of the customer's own strongest driver), because
  the median contribution is ~0.0005 while every customer's top one is ≥ 0.365 — "small"
  has no absolute meaning here. It never fires on the current test split; it exists so a
  featureless customer returns two drivers instead of a fabricated third. The guard is
  written `if not strongest > 0` rather than `== 0` because `max()` is NaN for an empty
  or all-NaN Series and NaN fails every comparison.
- **`score` is batch-first; `explain` rejects a batch** rather than silently explaining
  row 0. `Score` and `Driver` are frozen dataclasses shaped to map onto S4's Pydantic
  response models with no translation layer.

S6 decisions worth knowing before touching the `Dockerfile`:

- **`--no-deps` is load-bearing.** The runtime install filters `nvidia-` records out of
  the exported requirements; without `--no-deps` the installer re-resolves xgboost's
  dependency list and puts them straight back. The export is already the full closure.
- **The src layout is preserved inside `/var/task`.** `ARTIFACT_PATH` resolves relative
  to the package file, so `/var/task/src/churn/…` puts the model at `/var/task/artifacts`
  with no env override, and the container and a checkout agree on where it lives.
- **`lifespan="off"` in `lambda_handler.py`.** Lambda has no server lifecycle; the
  artifact warm-up is an import-time side effect in `api.py`, so it lands in the init
  phase. `tests/test_lambda_handler.py` asserts both halves in a subprocess, because
  import side effects happen once per interpreter and the test session has long since
  imported `churn.api`.

S7/S8 decisions worth knowing before touching `infra/bootstrap.yaml` or `template.yaml`:

- **The deploy role holds exactly one IAM action**, `iam:PassRole`, conditioned on
  `lambda.amazonaws.com`. That is why the Lambda *execution* role lives in
  `bootstrap.yaml` rather than in `template.yaml`: creating it from CI would need
  `iam:CreateRole` and `iam:AttachRolePolicy`, and a role that can mint roles can escalate
  to anything. `template.yaml` takes the ARN as a parameter and creates no IAM at all.
- **OIDC trust is `StringEquals` on the full subject**, and the subject carries **numeric
  IDs, not just names**: `repo:emdomingo@108572192/churn-risk-api@1344827975:ref:refs/heads/main`.
  Forks and pull requests present a different `sub` and are refused by STS, which is what
  makes the public repo safe to deploy from. A name-only policy fails closed and STS says
  only "Not authorized to perform sts:AssumeRoleWithWebIdentity" — CloudTrail's `userName`
  on the failed `AssumeRoleWithWebIdentity` event is where the real subject is legible.
  Because the IDs are immutable, renaming the repo no longer breaks the trust. See gate 13
  in `PLAN.md`.
- **ECR is IMMUTABLE and tags are commit SHAs.** A re-push of the same SHA is refused, so
  fixing a bad image means deleting the tag first. Lifecycle keeps the last 3.
- **`--provenance=false --sbom=false` is mandatory on every build.** Default buildx pushes
  an OCI *image index* with an `unknown/unknown` attestation manifest, and Lambda rejects
  that at `CreateFunction`. `deploy.yml` must carry both flags.
- **`sam deploy` is not used; `aws cloudformation deploy` is.** SAM's packaging step
  demands `--image-repository` and wants to build and push the image itself, which would
  bypass the SHA tagging and the provenance flags. The image is built and pushed by an
  explicit step, so the deploy is only a stack update — CloudFormation applies the SAM
  transform server-side.
- **The first request after a deploy returns 503.** This is known, measured, and
  deliberately not worked around — see gate 10 in `PLAN.md`. Init exceeds Lambda's 10s
  cap, gets suppressed and re-run inside the invocation, and 10 + 20.5s overruns API
  Gateway's 29s. Every subsequent request is 70–160ms. S10's smoke test has to tolerate
  it, and the README owes it an honest paragraph rather than a silent retry.

S9 decisions worth knowing before touching `.github/workflows/`:

- **`deploy.yml` calls `ci.yml` as a job**, via `workflow_call` with `build_image: false`.
  GitHub cannot express "job in workflow B needs a job in workflow A" — `workflow_run`
  fires after the fact and gates nothing — so `needs: ci` is what makes "a merge deploys
  only if CI is green" literally true. `build_image: false` stops the same commit being
  built twice.
- **CI's image build is path-filtered** to `Dockerfile`, `uv.lock`, `pyproject.toml`,
  `.dockerignore` — the whole break surface. An unknown base builds rather than skips.
  Gate 11 in `PLAN.md`.
- **`ANTHROPIC_API_KEY` is passed on every deploy** from a GitHub secret, reaching the
  command through `env:` rather than YAML interpolation. The `""` default is kept, so a
  keyless deploy still succeeds and `/recommend` degrades. Corollary: removing the secret
  *clears* the live key on the next deploy. Gate 12 in `PLAN.md`.
- **The deploy skips the build when the tag already exists.** ECR is IMMUTABLE, so a
  re-run on an unchanged commit would otherwise fail at the push instead of being a no-op.
- **The verification step retries `/health` up to 10 times**, tolerating the known 503
  from gate 10 and waiting for `model_version` to equal `0.1.0+<short sha>`.

**Two GitHub settings are required and are not in the repo:** the repository *variable*
`AWS_DEPLOY_ROLE_ARN` (the `DeployRoleArn` output of the bootstrap stack) and the
*secret* `ANTHROPIC_API_KEY`. Without the variable the deploy job cannot assume a role.

**S9 is written but unproven** — the workflows have never run. Proving them is a PR (CI
only, no AWS credentials touched) and a merge (deploy runs, live `model_version` changes).
Then **S10: README and smoke test**.
