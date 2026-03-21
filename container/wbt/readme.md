# image w/ WBT base

see Dockerfile [here](https://github.com/cefect/whitebox-python/blob/v2.4.0_dockerized/container/Dockerfile)


## explore the base
```bash
export IMAGE_NAME='cefect/wbt:v2.4.0.1'

#dump envs
env_name='wbt'
docker run --rm -v "$PWD/container/wbt:/out" $IMAGE_NAME bash -lc "conda run -n $env_name python -m pip freeze > /out/pip-freeze-$env_name.txt && conda env export -n $env_name > /out/conda-env-$env_name.lock.yml"

## Build Images: deploy
from WSL:
```bash

#set the image name and build
tag='v1.3'
export IMAGE_NAME='cefect/fdsc2:deploy-'$tag 
docker buildx build -f container/wbt/Dockerfile -t $IMAGE_NAME --target deploy .
 
 

# dump installed packages + conda env with one container invocation
docker run --rm -v "$PWD/container/wbt:/out" $IMAGE_NAME bash -lc "conda run -n deploy python -m pip freeze > /out/pip-freeze-deploy.txt && conda env export -n deploy > /out/conda-env-deploy.lock.yml"
```

## pcraster
```bash
target='pcraster'
export IMAGE_NAME="cefect/fdsc2:${target}-${tag}"
docker buildx build -f container/wbt/Dockerfile -t $IMAGE_NAME --target $target .


# dump installed packages + conda env with one container invocation
docker run --rm -v "$PWD/container/wbt:/out" $IMAGE_NAME bash -lc "conda run -n deploy python -m pip freeze > /out/pip-freeze-$target.txt && conda env export -n deploy > /out/conda-env-$target.lock.yml"



```
push to Docker Hub
```bash
# push
docker push $IMAGE_NAME

```

 
## Build Images: dev
from WSL
```bash
export IMAGE_NAME='cefect/fdsc2:dev-'$tag
docker buildx build -f container/wbt/Dockerfile -t $IMAGE_NAME --target dev .
```

 
update the .devcontainer/compose
```bash
 
yq -y -i '.services.dev.image = env.IMAGE_NAME' .devcontainer/docker-compose.yml
```
