# syntax=docker/dockerfile:1.7
FROM mambaorg/micromamba:1.5.8 AS deploy

# Keep builds reproducible-ish and quieter
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAMBA_NO_BANNER=1

ARG DEPLOY_ENV_NAME=deploy

WORKDIR /work

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        acl \
        ca-certificates \
        curl \
        bzip2 \
        xz-utils \
        tar \
        gzip \
        #procps \
        pciutils \
       # less \
       # strace \
       # gdb \
        binutils

USER $MAMBA_USER

# Create the deploy conda env from the numba-specific lock file in this repo.
COPY --chown=$MAMBA_USER:$MAMBA_USER container/numba/environment.yml /tmp/environment.yml

RUN --mount=type=cache,target=/tmp/.cache/pip,sharing=locked \
     micromamba create -y -n ${DEPLOY_ENV_NAME} -f /tmp/environment.yml

# Activate deploy by default (micromamba base image convention)
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=${DEPLOY_ENV_NAME}
ENV PATH=/opt/conda/envs/${DEPLOY_ENV_NAME}/bin:$PATH

# Deploy smoke test (kept minimal)
RUN python -c "import numpy as np; import numba; print('numpy', np.__version__, 'numba', numba.__version__)"

ENTRYPOINT []
CMD []

# DEV target -----------------------------
FROM deploy AS dev

# Clone deploy into a separate dev env and layer in notebook/debug extras.
ARG DEPLOY_ENV_NAME=deploy
ARG DEV_ENV_NAME=dev
COPY --chown=$MAMBA_USER:$MAMBA_USER container/numba/environment_dev.yml /tmp/environment_dev.yml

RUN --mount=type=cache,target=/tmp/.cache/pip,sharing=locked \
    micromamba create -y -n ${DEV_ENV_NAME} --clone ${DEPLOY_ENV_NAME} && \
    micromamba install -y -n ${DEV_ENV_NAME} -f /tmp/environment_dev.yml && \
    rm -f /tmp/environment_dev.yml

# Activate dev by default for the dev target.
ENV CONDA_DEFAULT_ENV=${DEV_ENV_NAME}
ENV PATH=/opt/conda/envs/${DEV_ENV_NAME}/bin:/opt/conda/envs/${DEPLOY_ENV_NAME}/bin:$PATH

# Dev smoke test: deploy packages are present and dev extras were layered on top.
RUN python -c "import numpy as np; import numba; import seaborn as sns; print('numpy', np.__version__, 'numba', numba.__version__, 'seaborn', sns.__version__)"

ENTRYPOINT []
CMD []
