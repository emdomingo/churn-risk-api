# Churn Risk API

A retention team can only call so many customers in a week. This service answers the three
questions that decide who gets called: **how likely is this customer to leave**, **what is
driving that**, and **what should we offer them**. It returns a calibrated probability (not
just a ranking, so "76%" means something a business can attach a budget to), the SHAP
drivers behind that number, and one retention action selected from a closed catalogue of
five. It runs as a container-image Lambda behind API Gateway, deployed from `main` by
GitHub Actions with no long-lived AWS keys anywhere.

**Live:** `https://03bmm97sm6.execute-api.ap-southeast-2.amazonaws.com`

> **The first request after a deploy returns 503.** This is measured and deliberate, not a
> flake — see [Cold start](#cold-start) below. Every request after it is 70–160 ms.

---

## Architecture

```mermaid
flowchart LR
    client([client]) --> apigw[API Gateway<br/>HTTP API]
    apigw --> lambda

    subgraph lambda [Lambda · container image · 3008 MB]
        direction TB
        mangum[Mangum] --> fastapi[FastAPI<br/>api.py]
        fastapi --> core

        subgraph core [pure core · no web, no AWS]
            direction TB
            scoring[scoring.py<br/>calibrated probability + band]
            explain[explain.py<br/>TreeExplainer → top-3 drivers]
            recommend[recommend.py<br/>select from catalogue]
        end
    end

    artifact[(model.joblib<br/>loaded at import)] -.-> core
    recommend -.->|band + 3 drivers<br/>+ catalogue only| anthropic([Anthropic API])
```

**Pure core, thin edges.** Every churn decision — cleaning, features, scoring, drivers,
action selection — is a plain function under `src/churn/` with no web framework and no
AWS. FastAPI, Mangum, and the Anthropic client are shells over that core. This is why the
test suite runs offline in seconds, and why the request path is short enough to whiteboard.

Two details that are easy to get wrong and matter here:

- **The artifact loads at module import, not per request**, so it lands in the Lambda init
  phase rather than taxing the first caller. (It has a cost — see [Cold start](#cold-start).)
- **The LLM sees only the risk band, the top three drivers, and the catalogue.** It never
  receives the customer record or the ID. It selects and justifies; it never invents an
  action or a customer fact, and its choice is validated against the catalogue before it
  reaches the response.

---

## The four routes

PowerShell aliases `curl` to `Invoke-WebRequest`, which takes neither `-X` nor `-d`. Use
`curl.exe` explicitly, as below, or run these from Git Bash.

```powershell
$API = "https://03bmm97sm6.execute-api.ap-southeast-2.amazonaws.com"
```

### `GET /health`

Readiness, not liveness — it depends on the artifact, so `ok` means the model actually
loaded. `model_version` is how you know which build is serving.

```powershell
curl.exe -s $API/health
```

```json
{"status":"ok","model_version":"0.1.0+3f5fb64"}
```

### `POST /score`

List in, list out, in request order — always, even for one customer. One `transform` and
one `predict_proba` for the whole batch; the route does not loop. Capped at 1000 records.

```powershell
curl.exe -s -X POST $API/score -H "Content-Type: application/json" -d '{\"customers\":[{\"customerID\":\"TEST-0001\",\"gender\":\"Female\",\"SeniorCitizen\":0,\"Partner\":\"No\",\"Dependents\":\"No\",\"tenure\":2,\"PhoneService\":\"Yes\",\"MultipleLines\":\"No\",\"InternetService\":\"Fiber optic\",\"OnlineSecurity\":\"No\",\"OnlineBackup\":\"No\",\"DeviceProtection\":\"No\",\"TechSupport\":\"No\",\"StreamingTV\":\"Yes\",\"StreamingMovies\":\"Yes\",\"Contract\":\"Month-to-month\",\"PaperlessBilling\":\"Yes\",\"PaymentMethod\":\"Electronic check\",\"MonthlyCharges\":98.7,\"TotalCharges\":197.4}]}'
```

```json
{
  "model_version": "0.1.0+3f5fb64",
  "results": [
    {"customerID": "TEST-0001", "probability": 0.7565723230441996, "band": "high"},
    {"customerID": "TEST-0002", "probability": 0.032667537551469394, "band": "low"}
  ]
}
```

### `POST /explain`

One customer, scored plus its top three drivers. Rejects a batch rather than silently
explaining row 0.

```powershell
curl.exe -s -X POST $API/explain -H "Content-Type: application/json" -d "@customer.json"
```

```json
{
  "model_version": "0.1.0+3f5fb64",
  "customerID": "TEST-0001",
  "probability": 0.7565723230441996,
  "band": "high",
  "drivers": [
    {"feature": "tenure",                  "value": 2,                "contribution": 0.5662, "direction": "increases"},
    {"feature": "MonthlyCharges",          "value": 98.7,             "contribution": 0.4265, "direction": "increases"},
    {"feature": "Contract=Month-to-month", "value": "Month-to-month", "contribution": 0.3947, "direction": "increases"}
  ]
}
```

Two things about drivers that look like bugs and are not:

- **`contribution` is in margin units, not probability**, and the values do not sum to the
  score. See [SHAP and calibration](#shap-explains-the-uncalibrated-margin).
- **A customer carries a contribution for levels they do not have.** One-hot encoding keeps
  every level, so `Contract=Two year` can appear as a *protective* driver for a
  month-to-month customer — the model is pricing its absence. `value` reports the
  customer's actual value on the source column, so a driver can legitimately read
  `feature: "Contract=Two year", value: "Month-to-month"`.

### `POST /recommend`

Score, drivers, and one action from the catalogue: `fee_waiver`, `plan_downgrade`,
`contract_term_incentive`, `support_callback`, `loyalty_perk`.

```powershell
curl.exe -s -X POST $API/recommend -H "Content-Type: application/json" -d "@customer.json"
```

```json
{
  "model_version": "0.1.0+3f5fb64",
  "customerID": "TEST-0001",
  "probability": 0.7565723230441996,
  "band": "high",
  "drivers": [ "…as above…" ],
  "action": "contract_term_incentive",
  "rationale": "Customer is on a month-to-month contract with high monthly charges and low tenure, so locking them into a discounted 1-2 year term directly addresses the top churn drivers.",
  "drivers_used": ["Contract=Month-to-month", "MonthlyCharges", "tenure"],
  "reason": null
}
```

**This route never returns 500 on an LLM failure.** If the call fails, times out, or no API
key is configured, it returns 200 with the score and drivers — the parts that need no
network — plus `action: null` and a `reason`. The status stays 200 because the request
*was* served: the caller got everything the service could compute.

**Why a closed catalogue rather than free text.** A retention offer is a commitment with a
cost. An LLM writing prose could invent "three months free" or promise a discount nobody
approved. Restricting it to five identifiers turns generation into selection: auditable
(every recommendation is one of five known values in a log line), bounded (finance signs
off once), testable (assert nothing else ever escapes — impossible against free text), and
degradable (an off-list answer is detectable, so the failure is "no action" rather than "an
action nobody authorised").

---

## Metrics

Test split, 1409 rows, seed 42. Base churn rate 26.5%.

| Metric | Test | Why this metric |
|---|---|---|
| ROC-AUC | **0.833** | Ranking quality across all thresholds. Reported because it is the common currency — but on a 26% positive class it flatters, which is why it is not the headline. |
| PR-AUC | **0.638** | Ranking quality on the class that matters. No-skill here is 0.265, so this is the honest read of "how well do we find churners". |
| Brier | **0.141** | Whether the scores are *probabilities*, not just an ordering. Uncalibrated: 0.168. This is the number that justifies attaching a budget to "76%". |
| tenure-only baseline | 0.734 | The floor. A single feature gets you most of the way; the model has to beat it to be worth deploying. |
| ROC-AUC, train | 0.870 | Overfit check. The 0.037 gap to test is the memorisation. |

Accuracy is deliberately absent: predicting "nobody churns" scores 73.5% on this data and
is worthless.

**Confusion matrix at p ≥ 0.167:**

|  | predicted stay | predicted churn |
|---|---|---|
| **actually stayed** | 686 | 349 |
| **actually churned** | 51 | 323 |

Recall 0.864, precision 0.481, flagging 47.7% of the base.

**The threshold is derived, not picked.** `MISSED_CHURNER_COST_RATIO = 5.0` — one missed
churner costs about five wasted retention offers — and the threshold falls out as
`1 / (1 + 5) = 0.167`. Changing the business assumption changes the threshold; nobody
tunes the number directly. That same cut *is* the low/medium band boundary, so "not low"
and "the model says intervene" are one predicate, and the matrix above describes exactly
the medium+high population. A test fails if the two ever drift apart.

### Calibration

**Sigmoid, not isotonic.** ~1,400 validation rows is too few for isotonic's staircase to
be anything but overfitting. The distortion being corrected comes from
`scale_pos_weight`, which is close to a constant shift in log-odds — exactly what a
two-parameter sigmoid undoes.

**The validation split is touched by nothing but the calibrator.** Tree count and depth come
from a 20% stopping slice carved out of *train*, never from `val`. A fixed 300-tree budget
was tried first and measurably overfit.

**The booster is refit at the chosen count.** Early stopping leaves unused trees in the
model and `TreeExplainer` would read them, so training runs twice on purpose. A test
asserts SHAP additivity against the shipped booster's margin.

### Imbalance

`scale_pos_weight = 2.769` (the ratio of negatives to positives), not resampling. It
reweights the objective without inventing or discarding rows, and its distortion is
correctable — which is what the calibration step then does. The cost of the choice is that
raw model output is not a probability until calibrated, which is precisely why Brier is in
the table above.

### SHAP explains the uncalibrated margin

`CalibratedClassifierCV` wraps the booster; `TreeExplainer` runs on the booster underneath.
**So the returned score is calibrated while the drivers explain the pre-calibration
margin.** Contributions are in margin units and do not sum to the probability.

This is defensible rather than ideal: calibration is a monotone map, so the *ranking* of
drivers is unaffected, and ranking is what a driver list is for. But it is a real seam and
it is stated here rather than glossed. The honest alternative — explaining the calibrated
output directly — needs a model-agnostic explainer (KernelSHAP), which is orders of
magnitude slower per request and would not fit a per-request `/explain`.

The global top 10 by mean |SHAP| matches the univariate prior registered in
`exploration.ipynb` *before* the model was fitted, and contains none of `gender`,
`PhoneService`, or `MultipleLines`:

`Contract=Month-to-month` · `tenure` · `InternetService=Fiber optic` · `MonthlyCharges` ·
`OnlineSecurity=No` · `Contract=Two year` · `TechSupport=No` ·
`PaymentMethod=Electronic check` · `PaperlessBilling=No` · `TotalCharges`

---

## Cold start

**The first request after a deploy returns 503.** Measured, understood, and deliberately
not worked around:

1. Init imports pandas, sklearn, shap, and xgboost off a streamed 1.28 GB container image
   and loads the artifact. That overruns Lambda's 10s init cap.
2. Lambda suppresses the timeout and re-runs init *inside* the invocation — about 20.5s.
3. 10 + 20.5s overruns API Gateway's 29s ceiling. The caller gets a 503.

Every subsequent request is 70–160 ms.

The fix would be to defer the heavy imports out of module scope so init stays trivial and
the ~20s lands inside the invocation budget — converting the 503 into a slow 200. That
inverts the "artifact loads at import so it lands in the init phase" decision, and the
tradeoff was taken knowingly. A warmer (EventBridge pinging `/health`) papers over the
cause; provisioned concurrency costs ~$19–32/month and ends the free-tier story. The
deploy pipeline and the smoke test both retry through it rather than hiding it.

The image is 1.28 GB because `shap`, `xgboost`, `sklearn`, and `pandas` are all genuinely
needed at inference. It would be 2.1 GB without a filter that drops the `nvidia-` CUDA
packages xgboost pulls in but never uses on Lambda. Measured trimming beyond that returns
~75 MB for real fragility, so it stopped there.

---

## Running it

```powershell
uv sync --all-groups                        # install
uv run pytest                               # 142 tests, offline, no AWS
uv run ruff check .                         # lint
uv run --group train python -m churn.train  # train → calibrate → SHAP → artifacts/
uv run uvicorn churn.api:app --reload       # serve locally on :8000
docker build -t churn:local .               # trains in stage 1; ~1.28 GB
```

The suite never loads a prebuilt model — `tests/conftest.py` trains a tiny one on a
committed 200-row fixture at session scope. That is what lets CI run with no AWS, no
network, and no artifact in git.

**Smoke test against the live URL** — the one test that touches the network, opt-in so the
default suite stays offline:

```powershell
$env:CHURN_SMOKE_URL = "https://03bmm97sm6.execute-api.ap-southeast-2.amazonaws.com"
uv run pytest tests/test_smoke.py -v
```

Set `CHURN_SMOKE_VERSION` too to assert a specific build is serving.

---

## Deployment

Push to `main` → CI (ruff + pytest) → build → push to ECR → CloudFormation → verify.

- **No long-lived AWS keys.** GitHub Actions presents an OIDC token; the trust policy is
  `StringEquals` on the full subject, including GitHub's immutable numeric owner and repo
  IDs. Forks and pull requests present a different subject and are refused by STS — which
  is what makes a public repo safe to deploy from.
- **The deploy role holds exactly one IAM action**, `iam:PassRole`, conditioned on
  `lambda.amazonaws.com`. The Lambda execution role is created by the one-time bootstrap
  stack, not by CI: creating it from CI would need `iam:CreateRole`, and a role that can
  mint roles can escalate to anything.
- **ECR is immutable and tags are commit SHAs**, so one tag is always exactly one build.
  `model_version` in every response is that SHA — the deploy is verified by polling
  `/health` until it reports the new one.
- **Two settings live outside the repo** and are worth knowing about: the repository
  variable `AWS_DEPLOY_ROLE_ARN` and the secret `ANTHROPIC_API_KEY`. Without the key the
  service still deploys and `/recommend` degrades to `action: null`.

---

## Notable choices

- **The training CSV is committed** (~1 MB in `data/`). The project's own guardrails say
  not to commit data; at 1 MB the reasoning behind that rule doesn't reach, and committing
  it makes the Docker build hermetic — no external host on the deploy path, and the exact
  bytes trained on are in the repo. The rule still holds for the model artifact, which is
  gitignored and never enters git. Deliberate exception, recorded as one.
- **The base image is `python:3.11-slim-bookworm`, not the AWS Lambda base.** The AWS
  Python 3.11 image is Amazon Linux 2 (glibc 2.26) and xgboost ships `manylinux_2_28`
  wheels only, so nothing installs there without compiling. `awslambdaric` is wired in by
  hand as a consequence.
- **`TotalCharges` blanks are filled with 0.0**, not imputed. All 11 are `tenure == 0` —
  brand-new customers, correctly billed zero. Imputing a mean would invent history for the
  customers most worth scoring.
- **`--provenance=false --sbom=false` on every image build.** Default buildx pushes an OCI
  image index with an `unknown/unknown` attestation manifest, which Lambda rejects at
  `CreateFunction` with an error that does not say why.

---

## What I'd do next

**Monitoring.** The structured JSON logs already carry request ID, latency, and score, so
the first additions are cheap: a CloudWatch metric filter on the score distribution and on
the `/recommend` degradation rate. The second is the one that matters — a rising
`action: null` rate is an LLM outage nobody would otherwise notice, because the route
returns 200 by design.

**Drift.** Nothing watches for it today. The cheapest useful version is a scheduled job
comparing the live score distribution against the test-split distribution (PSI on the
score, plus per-feature PSI on the top five drivers). Feature drift on `Contract` or
`tenure` would move the score distribution before any label arrives, which is the only
early signal available when churn labels lag by a billing cycle or more.

**Retraining.** Quarterly is the right cadence for a churn model on monthly contracts —
frequent enough to track pricing and plan changes, infrequent enough that each retrain has
genuinely new labels. Retraining is already a `docker build`, so the work is a scheduled
pipeline plus a gate: ship only if PR-AUC on a held-out recent slice beats the incumbent.
The calibration step needs redoing every time, not just the booster.

**Auth and rate limiting.** The live URL is open, which was fine while the function ran
without an LLM key and is a real gap now that `/recommend` costs money per call. An API
Gateway usage plan with a key, or IAM auth, is the small version.

**Threshold as configuration.** `MISSED_CHURNER_COST_RATIO` is a business assumption
compiled into the image. It belongs in the stack parameters so retention can revise its own
economics without a rebuild.

**A held-out temporal split.** The current split is random, which leaks calendar time
across train and test. A time-ordered split would give an honest estimate of how the model
performs on the customers it will actually see next.
