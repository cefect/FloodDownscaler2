# Container for numba experiments

TODO: run the experiments and test this. 

## Build image (deploy target)
From repo root:

```bash
tag="v0.1"
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
tag="v0.1"
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
docker run --rm -v "$PWD/container/numba:/out" "${IMAGE_NAME_DEPLOY}" bash -lc \
  "micromamba run -n deploy python -m pip freeze > /out/pip-freeze-deploy.txt && \
   micromamba env export -n deploy > /out/conda-env-deploy.lock.yml"
```

## Dump installed packages for dev env
The `dev` target creates a second conda env `dev` by cloning `deploy`, then applies `container/numba/environment_dev.yml`.

```bash
docker run --rm -v "$PWD/container/numba:/out" "${IMAGE_NAME_DEV}" bash -lc \
  "micromamba run -n dev python -m pip freeze > /out/pip-freeze-dev.txt && \
   micromamba env export -n dev > /out/conda-env-dev.lock.yml"
```
