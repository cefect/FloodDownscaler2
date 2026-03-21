# US Flood Events dev container


## BASE IMAGE=====================

```bash
#base image
docker run -it --rm condaforge/miniforge3:25.3.1-0 bash

 
 
```
 


## Build Images: deploy
from WSL:
```bash

#set the image name
IMAGE_NAME='cefect/fdsc2:deploy-v0.6'

# build the container
docker buildx build -f container/Dockerfile -t $IMAGE_NAME --target deploy .
 
 

# dump installed packages + conda env with one container invocation
docker run --rm -v "$PWD/container:/out" $IMAGE_NAME bash -lc "conda run -n deploy python -m pip freeze > /out/pip-freeze-deploy.txt && conda env export -n deploy > /out/conda-env-deploy.lock.yml"
```

push to Docker Hub
```bash
# push
docker push $IMAGE_NAME

```

 
## Build Images: dev
from WSL
```bash
export IMAGE_NAME='cefect/fdsc2:dev-v0.8'
docker buildx build -f container/Dockerfile -t $IMAGE_NAME --target dev .
```

 
update the .devcontainer/compose
```bash
 
yq -y -i '.services.dev.image = env.IMAGE_NAME' .devcontainer/docker-compose.yml
```
