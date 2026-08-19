# SPEC — Churn-Risk Scoring Microservice with Explainable, Actionable Predictions

> **Portfolio project.** Three capabilities in one service: churn prediction, explainability, and next-best-action, deployed serverless on AWS. Time budget 2–3 nights. Deployment and explainability are the point; the model is deliberately simple.

> **Working mode.** This is built with Claude Code. Claude Code writes the code; **I own the understanding** — every architectural decision, every metric, every deploy step should be explainable by me before it's accepted. When a section below says *"I must be able to explain X"*, that's a gate, not a nice-to-have.

---

## 1. Goal & success criteria

Build and deploy a service that, given a customer record, returns:
1. a **churn risk score** (calibrated probability),
2. the **top drivers** of that score (SHAP), and
3. a **recommended retention action** (LLM, grounded in the drivers).

**Done when:**
- `/score`, `/explain`, `/recommend` return correct responses from a **live AWS Lambda + API Gateway URL**.
- CI/CD pipeline (GitHub Actions) tests → builds container image → pushes to ECR → deploys, on push to `main`.
- README lets a stranger hit the live endpoint and understand the design in <5 min.
- I can whiteboard the full request path and the model's top 3 churn drivers from memory.

**Explicit non-goals:** SOTA accuracy, hyperparameter grids, a polished frontend, multi-model ensembles, real-time streaming. Resist scope creep toward any of these.

---

## 2. Dataset

**Telco Customer Churn** (IBM sample, ~7,043 rows, 21 columns, binary `Churn`; ~26.5% positive).

- **Source (no auth, no Kaggle):** pull directly from IBM's own repo —
  `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
  **Amended (S0):** the full CSV (~1 MB) is downloaded once and **committed** to `data/`. No fetch step, no pinned SHA, no network during the Docker build or the test suite. Rationale: the image is rebuilt on every merge to `main`, so a build-time fetch would put an external host on the deploy path — and §10 says deploy is the thing that must never break. Committing also makes reproducibility stronger, not weaker: the exact bytes trained on are in the repo.
- Commit a small sample (~200 rows) as a test fixture.
- Known gotchas to handle explicitly (Claude Code should not silently paper over these):
  - `TotalCharges` is object-typed with blank strings for new customers → coerce, decide impute vs drop, **document the choice**.
  - Class imbalance (~26% churn) → handle via `scale_pos_weight` / class weighting, not resampling. Report on it.
- **I must be able to explain:** what each of the top-10 features means and why it plausibly relates to churn.

---

## 3. Model

- **Single gradient-boosted tree** (XGBoost or LightGBM). No ensembles, no stacking.
- Train/val/test split with a fixed seed. Stratified.
- **Calibrate** probabilities (`CalibratedClassifierCV` or isotonic) — a churn *score* that isn't calibrated is misleading, and calibration is a strong talking point.
- Persist model + preprocessing as a single artifact (pin versions; `joblib` or native booster format).
- Metrics to report in README: ROC-AUC, PR-AUC, Brier score (for calibration), and a confusion matrix at a chosen threshold. **Explain why PR-AUC and Brier matter here, not just accuracy.**
- **SHAP** `TreeExplainer` fit at build time; explainer persisted alongside the model.

**I must be able to explain:** why a tree model over logistic regression here; what a SHAP value actually represents; why calibration ≠ accuracy.

---

## 4. API (FastAPI, packaged for Lambda)

Framework: **FastAPI + Mangum** adapter (Mangum wraps ASGI for Lambda). Served behind API Gateway (HTTP API).

Endpoints:

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/health` | GET | — | `{status, model_version}` |
| `/score` | POST | one or many customer records | risk score(s) + risk band (low/med/high) |
| `/explain` | POST | one customer record | score + top-N SHAP drivers (feature, value, contribution, direction) |
| `/recommend` | POST | one customer record | score + drivers + LLM-drafted retention action |

- **Pydantic** models for request/response — schema is the contract. Validate inputs strictly.
- Batch support on `/score` (list in, list out) — shows you think about throughput.
- Version the model in every response payload.

**I must be able to explain:** why Mangum, what cold starts are, and what API Gateway adds in front of Lambda.

---

## 5. `/recommend` — the agentic / next-best-action layer

- Takes the churn score + SHAP drivers, calls an LLM (Anthropic API) to draft **one concrete retention action** grounded in the drivers.
- The prompt must **only** reason over the structured drivers passed in — no hallucinated customer facts. Feed it: risk band, top 3 drivers with directions, and a fixed catalogue of allowed actions (fee waiver, plan downgrade offer, contract-term incentive, support callback, loyalty perk). LLM **selects and justifies**, does not invent actions.
- Output: `{action, rationale, drivers_used}`. Constrained and auditable by design: every recommendation traces back to the specific drivers that produced it.
- API key via environment variable / Lambda env, **never committed**. Document local `.env` handling.
- Graceful degradation: if the LLM call fails, return score + drivers with `action: null` and a reason. The service must not 500 because the LLM is down.

**I must be able to explain:** why the action set is a closed catalogue (safety/auditability), and how this differs from letting the LLM freeform.

---

## 6. Packaging & deploy (AWS Lambda + API Gateway, container image)

- **Container image Lambda** (not zip) — model + SHAP + deps exceed the zip limit comfortably, and it makes the CI/CD cleaner.
- Base: AWS Lambda Python base image. Multi-stage Dockerfile to keep image lean.
- Infra as code: prefer **AWS SAM** (`template.yaml`) or Terraform — pick one, document why. SAM is lighter for a single Lambda + HTTP API.
- Resources: one Lambda (container), one HTTP API Gateway, one ECR repo. Free-tier conscious — note expected cost (~$0).
- Config via Lambda env vars: `MODEL_VERSION`, `ANTHROPIC_API_KEY`, `LOG_LEVEL`.
- Structured JSON logging to CloudWatch. Log request id, latency, score — not raw PII.

**I must be able to explain:** the full cold-path request lifecycle (client → API GW → Lambda cold start → Mangum → FastAPI → response), and why container-image Lambda over zip.

---

## 7. CI/CD (GitHub Actions)

Two workflows:
- **`ci.yml`** (on PR): lint (ruff), type-check (optional), run pytest against the sample fixture. No AWS.
- **`deploy.yml`** (on push to `main`): build image → push to ECR → `sam deploy` (or Terraform apply). Auth to AWS via **OIDC role**, not long-lived keys — call this out as a deliberate security choice.

Guard: deploy only runs if CI passes.

**I must be able to explain:** why OIDC over stored AWS keys, and what each pipeline stage does.

---

## 8. Tests (thin but real)

- Unit: preprocessing handles the `TotalCharges` blanks; scorer returns a probability in [0,1]; risk-band boundaries.
- Contract: each endpoint against Pydantic schemas using the 200-row fixture.
- `/recommend`: mock the LLM — assert it only ever returns an action from the allowed catalogue and degrades gracefully on failure.
- One end-to-end smoke test hittable against the deployed URL (manual or a final CI step).

---

## 9. README (the portfolio surface)

Must contain: one-paragraph problem framing tied to churn/retention; architecture diagram (client → API GW → Lambda → model/SHAP/LLM); live endpoint + `curl` examples for all four routes; metrics table with a sentence on why each metric; the calibration and imbalance decisions; a "what I'd do next" section (monitoring, drift, retraining cadence). Keep prose tight.

---

## 10. Suggested night plan (guide, not contract)

- **Night 1 — model + explainability.** Data handling, train GBT, calibrate, fit SHAP, persist artifacts, report metrics. FastAPI runs locally with `/score` + `/explain`.
- **Night 2 — service + agent + container.** Add `/recommend` with the constrained LLM layer. Pydantic contracts, tests, Dockerfile, Mangum. Runs in a local container.
- **Night 3 — deploy.** SAM template, ECR, live Lambda + API Gateway, GitHub Actions with OIDC, README. Smoke-test the live URL.

If time runs short, drop batch-scoring and the "what I'd do next" section first — never drop deploy or calibration; those are the differentiators.

---

## 11. Guardrails for Claude Code

- Do not add dependencies not needed for a listed feature.
- Do not upgrade the model beyond a single GBT.
- Do not commit: the model artifact (>a few MB), any secret, or `.env`. (**Amended (S0):** the ~1 MB training CSV *is* committed — see §2. The guardrail still holds for the model artifact, which is where it bites.)
- When a design choice has a tradeoff (impute vs drop, SAM vs Terraform, threshold value), **stop and surface it to me** rather than silently choosing — I own those decisions.
- Pin all versions. Reproducibility is part of the grade.
