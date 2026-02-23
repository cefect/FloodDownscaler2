# Container for numba experiments

TODO: run the experiments and test this.

## Build image (deploy target)
From repo root:

```bash
tag="v0.2"
export IMAGE_NAME_DEPLOY="cefect/fdsc2:numba-deploy-${tag}"

docker buildx build \
  -f container/numba/Dockerfile \
  --target deploy \
  -t "${IMAGE_NAME_DEPLOY}" \
  --load \
  .
```

## Build image (dev target)
From repo root:

```bash
tag="v0.2"
export IMAGE_NAME_DEV="cefect/fdsc2:numba-dev-${tag}"

docker buildx build \
  -f container/numba/Dockerfile \
  --target dev \
  -t "${IMAGE_NAME_DEV}" \
  --load \
  .
```

## Dump installed packages for deploy env
The `deploy` target creates conda env `deploy` from `container/numba/environment.yml`.

```bash
docker run --rm -u "$(id -u)":"$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -v "$PWD/container/numba:/out" "${IMAGE_NAME_DEPLOY}" bash -lc \
  "conda run -n deploy python -m pip freeze > /out/pip-freeze-deploy.txt && \
   conda env export -n deploy > /out/conda-env-deploy.lock.yml"
```

## Dump installed packages for dev env
The `dev` target creates a second conda env `dev` by cloning `deploy`, then installs notebook/debug extras in that env.

```bash
docker run --rm -u "$(id -u)":"$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -v "$PWD/container/numba:/out" "${IMAGE_NAME_DEV}" bash -lc \
  "conda run -n dev python -m pip freeze > /out/pip-freeze-dev.txt && \
   conda env export -n dev > /out/conda-env-dev.lock.yml"
```
