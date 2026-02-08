# US Flood Events dev container


## BASE IMAGE=====================

```bash
#base image
docker run -it --rm condaforge/miniforge3:25.3.1-0 bash

 
 
```

### properties
```bash

#comes with uid1000
(base) root@1a4ddbd605e6:/# getent passwd
    root:x:0:0:root:/root:/bin/bash
    daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
    bin:x:2:2:bin:/bin:/usr/sbin/nologin
    sys:x:3:3:sys:/dev:/usr/sbin/nologin
    sync:x:4:65534:sync:/bin:/bin/sync
    games:x:5:60:games:/usr/games:/usr/sbin/nologin
    man:x:6:12:man:/var/cache/man:/usr/sbin/nologin
    lp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologin
    mail:x:8:8:mail:/var/mail:/usr/sbin/nologin
    news:x:9:9:news:/var/spool/news:/usr/sbin/nologin
    uucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologin
    proxy:x:13:13:proxy:/bin:/usr/sbin/nologin
    www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin
    backup:x:34:34:backup:/var/backups:/usr/sbin/nologin
    list:x:38:38:Mailing List Manager:/var/list:/usr/sbin/nologin
    irc:x:39:39:ircd:/run/ircd:/usr/sbin/nologin
    _apt:x:42:65534::/nonexistent:/usr/sbin/nologin
    nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin
    ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash

#base conda is activated
(base) root@1a4ddbd605e6:/# cat ~/.bashrc
    . /opt/conda/etc/profile.d/conda.sh && conda activate base

(base) root@1a4ddbd605e6:/# conda env list

    # conda environments:
    #
    base                 * /opt/conda

(base) root@1a4ddbd605e6:/# echo $PATH
    /opt/conda/bin:/opt/conda/condabin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```




## Build Images: deploy
from WSL:
```bash

#set the image name
IMAGE_NAME='cefect/fdsc2:deploy-v0.2'

# build the container
docker build -f container/Dockerfile -t $IMAGE_NAME --target deploy .

```


explore w/ a random user
```bash
echo $IMAGE_NAME

docker run --user 12345:12345 -it --rm $IMAGE_NAME bash
docker run -it --rm $IMAGE_NAME bash

# dump installed packages
docker run --rm $IMAGE_NAME conda run -n deploy python -m pip freeze > container/pip-freeze-deploy.txt

#conda dump
docker run --rm $IMAGE_NAME conda env export -n deploy > container/conda-env-deploy.yml
```

push to Docker Hub
```bash
# push
docker push $IMAGE_NAME

```

 
## Build Images: dev
from WSL
```bash
IMAGE_NAME='cefect/fdsc2:dev-v0.2'
docker build -f container/Dockerfile -t $IMAGE_NAME --target dev .
```

 
update devcontainer docker-compose.yml to use new image

see `.devcontainer/readme.md`
 
