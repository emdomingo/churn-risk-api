# Multi-stage: stage 1 trains the model, stage 2 serves it. The trainer is discarded, so
# matplotlib, the training CSV and the whole uv toolchain stay out of the shipped image.
#
# The build fetches nothing but Python packages: `data/Telco-Customer-Churn.csv` is
# committed for exactly this reason (SPEC.md §11, amended). No S3, no dataset host, so a
# build is reproducible from the git tree alone.
#
# Both stages start from the same image on purpose. The artifact is a joblib pickle, only
# loadable by the library versions that wrote it; training and serving on one interpreter
# and one `uv.lock` makes writer/reader drift impossible rather than merely unlikely.
#
# **Why not `public.ecr.aws/lambda/python:3.11` (S6 gate).** That base is Amazon Linux 2,
# glibc 2.26. xgboost publishes linux wheels as `manylinux_2_28` only -- for 3.2.0 and for
# every later release -- so pip finds no compatible wheel there and falls back to the
# sdist, which needs a C++ toolchain and cmake. The alternatives were to compile xgboost
# on every build, or to move to the AL2023-based 3.13 image and take xgboost 3.4.1, which
# would invalidate the recorded metrics. Debian slim (glibc 2.36) keeps the locked
# versions and the measured numbers; the cost is wiring the Lambda Runtime Interface
# Client by hand, which is the four lines at the bottom of this file. AWS documents this
# path for exactly this case.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.11.1

# Pinned to the uv that wrote `uv.lock`; `--frozen` below then means the same resolver
# reads it, not merely a compatible one.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --------------------------------------------------------------------------- trainer --
FROM python:${PYTHON_VERSION}-slim-bookworm AS trainer

COPY --from=uv /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/build/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Dependencies before source: this layer is the expensive one (xgboost, shap, sklearn)
# and it rebuilds only when the lock changes, not when a docstring does.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --group train

COPY data/ ./data/

# The commit being built, injected because `.git` never enters the image. Unset locally,
# which lands `resolve_model_version` on its package-version fallback -- fine for a local
# build, while CI passes the real sha so the live `model_version` moves on every merge.
ARG MODEL_VERSION=""
ENV MODEL_VERSION=${MODEL_VERSION}

RUN uv run --frozen --no-dev --group train python -m churn.train

# --------------------------------------------------------------------------- runtime --
FROM python:${PYTHON_VERSION}-slim-bookworm

COPY --from=uv /uv /bin/uv

ENV LAMBDA_TASK_ROOT=/var/task
WORKDIR ${LAMBDA_TASK_ROOT}

# Runtime deps only -- no dev, no train group, and not the `churn` package itself, whose
# source is copied in below. Installed into the image's own interpreter rather than a
# virtualenv because `awslambdaric` is executed as `python -m` by the entrypoint and must
# share an interpreter with the application it imports.
#
# **The `nvidia-` filter (S6 gate).** xgboost declares `nvidia-nccl-cu12` on linux for
# distributed GPU training. This function does CPU inference on a booster trained CPU-only,
# so those libraries are never dlopen'd -- verified by deleting them from a built image and
# re-invoking: `/score` and `/explain` returned byte-identical responses. Keeping them costs
# 454MB in the layer AWS pulls on a cold start, which is this design's known weak point.
# Filtering at install also keeps 326MB off every build.
#
# The lock stays untouched, so the image is a deliberate subset of the resolution rather
# than a different resolution. If xgboost ever moves nccl onto an import-time path, the
# smoke test is what catches it. The awk is line-oriented because `uv export` writes each
# requirement as a record with its `--hash` lines indented beneath it; dropping only the
# `nvidia-` line would leave orphaned hashes that the installer rejects. `--no-deps` is
# what makes the filter stick: without it the installer re-resolves xgboost's dependency
# list and pulls nccl straight back in. The exported file is already the complete closure,
# so there is nothing left for a resolver to add.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/requirements.txt \
    && awk '/^[^[:space:]]/ { skip = ($0 ~ /^nvidia-/) } !skip' /tmp/requirements.txt > /tmp/requirements.cpu.txt \
    && uv pip install --system --no-cache --no-deps --compile-bytecode -r /tmp/requirements.cpu.txt \
    && rm -f /tmp/requirements.txt /tmp/requirements.cpu.txt pyproject.toml uv.lock /bin/uv

# The repo's src layout is preserved inside the task root, so `artifact.ARTIFACT_PATH` --
# which resolves relative to the package file -- lands on /var/task/artifacts with no
# environment override. The container and a local checkout agree on where the model is.
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY --from=trainer /build/artifacts/model.joblib ${LAMBDA_TASK_ROOT}/artifacts/model.joblib

# Compile our own source too. Between this and `--compile-bytecode` above, every module
# the function imports has a `.pyc` beside it before the image is ever pulled.
RUN python -m compileall -q ${LAMBDA_TASK_ROOT}/src

# **Precompiled bytecode is a cold-start decision, measured (S8).** A Lambda filesystem is
# read-only, so Python can never cache bytecode at runtime: with no `.pyc` in the image,
# every cold start recompiles pandas, sklearn, shap and xgboost from source, and pays it
# again on the next one. Measured in this image: 8.53s to import with no cache against
# 2.40s with one. On Lambda, where image layers are streamed on first touch, the uncached
# path overran the 10s init limit outright and the first deploy answered 503.
#
# The cost is ~150MB of `.pyc`. That is the right side of the trade: bytecode is faulted
# in lazily like everything else, and only for modules actually imported.
#
# PYTHONDONTWRITEBYTECODE stays set precisely *because* everything is compiled already --
# it stops the interpreter attempting writes the read-only filesystem would refuse.
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# What the AWS base image would have provided. `awslambdaric` is the runtime interface
# client: it polls the Lambda Runtime API for an event, imports the handler, calls it, and
# posts the response back. Importing the handler is what loads the artifact, and that
# import happens during the init phase -- so the model is resident before the first
# request rather than on it.
#
# The Runtime Interface Emulator is deliberately NOT baked in. It is a local testing tool,
# and downloading it at build time would put a GitHub release on the path of a production
# image build. `docker run` mounts it from the host instead:
#
#   docker run -p 9000:8080 -v $HOME/.aws-lambda-rie:/aws-lambda #     --entrypoint /aws-lambda/aws-lambda-rie churn:local #     python -m awslambdaric churn.lambda_handler.handler
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["churn.lambda_handler.handler"]
