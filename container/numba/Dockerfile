# syntax=docker/dockerfile:1.7
FROM condaforge/miniforge3:24.11.3-0 AS deploy

SHELL ["/bin/bash", "-o", "pipefail", "-lc"]

# Keep builds reproducible-ish and quieter
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG DEPLOY_ENV_NAME=deploy

WORKDIR /work

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

# Create the deploy conda env from the numba-specific lock file in this repo.
COPY container/numba/environment.yml /tmp/environment.yml

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    conda config --set channel_priority strict && \
    conda env create -y -n ${DEPLOY_ENV_NAME} -f /tmp/environment.yml && \
    conda clean -afy && \
    rm -f /tmp/environment.yml

# Activate deploy by default.
ENV CONDA_DEFAULT_ENV=${DEPLOY_ENV_NAME}
ENV PATH=/opt/conda/envs/${DEPLOY_ENV_NAME}/bin:/opt/conda/condabin:$PATH

# Deploy smoke test (kept minimal)
RUN python -c "import numpy as np; import numba; print('numpy', np.__version__, 'numba', numba.__version__)"

ENTRYPOINT []
CMD []

# DEV target -----------------------------
FROM deploy AS dev

# Clone deploy into a separate dev env and layer in notebook/debug extras.
ARG DEPLOY_ENV_NAME=deploy
ARG DEV_ENV_NAME=dev

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    conda create -y -n ${DEV_ENV_NAME} --clone ${DEPLOY_ENV_NAME} && \
    mamba install -y --strict-channel-priority --override-channels -c conda-forge -n ${DEV_ENV_NAME} \
        jupyterlab notebook pyproj shapely ipykernel contextily nbdime seaborn imageio matplotlib colorcet openpyxl && \
    conda run -n ${DEV_ENV_NAME} python -m pip install --no-cache-dir palettable beautifulsoup4 memory-profiler

# Activate dev by default for the dev target.
ENV CONDA_DEFAULT_ENV=${DEV_ENV_NAME}
ENV PATH=/opt/conda/envs/${DEV_ENV_NAME}/bin:/opt/conda/envs/${DEPLOY_ENV_NAME}/bin:/opt/conda/condabin:$PATH

# Dev smoke test: deploy packages are present and dev extras were layered on top.
RUN python -c "import numpy as np; import numba; import seaborn as sns; print('numpy', np.__version__, 'numba', numba.__version__, 'seaborn', sns.__version__)"

ENTRYPOINT []
CMD []
