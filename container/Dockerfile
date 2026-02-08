FROM condaforge/miniforge3:25.3.1-0 AS base
#https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
# base environment is pre-activated
# these are pre-isntalled: wget bzip2 ca-certificates, git, tini

# Use bash for RUN and fail on pipe errors (good practice)
SHELL ["/bin/bash", "-o", "pipefail", "-lc"]

 








#make permissive to any non-root user
ARG TARGET_DIRS="/workspace /opt/conda"


# ---- basic tools for HTC runs + permission tooling
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    #--mount=type=cache,target=/var/lib/apt,sharing=locked \
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
        binutils \
    && rm -rf /var/lib/apt/lists/*

# set permissions on target dirs for arbitrary users
RUN for dir in $TARGET_DIRS; do \
        install -dm0755 "$dir" \
        && setfacl -b "$dir" \
        && setfacl -m u::rwx,g::rwx,o::rwx "$dir" \
        && setfacl -d -m u::rwx,g::rwx,o::rwx "$dir"; \
    done 

# Set a permissive umask so files created inside the container are world-writable.
RUN echo 'umask 000' >> /etc/bash.bashrc && \
    echo 'umask 000' >> /etc/profile 




# ---- 2) Build environments as root (fast, cacheable)
COPY container/environment.yml /tmp/environment.yml

ARG env_name=deploy

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    conda config --set channel_priority strict && \
    mamba env create -yv --strict-channel-priority --name ${env_name} --file /tmp/environment.yml && \
    rm -f /tmp/environment.yml


# activate the environment by default for all users
# the miniconda container activates base by default, here we change that to our env
RUN cat <<EOF >/etc/profile.d/conda_env.sh
# Activate the env for all login shells.
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  . "${CONDA_DIR}/etc/profile.d/conda.sh"
  if [ "\${CONDA_DEFAULT_ENV:-}" != "${env_name}" ]; then
    conda activate "${env_name}"
  fi
fi
EOF
RUN echo 'if [ -f /etc/profile.d/conda_env.sh ]; then . /etc/profile.d/conda_env.sh; fi' >> /etc/bash.bashrc




# set some base directory env vars helpful for random users
ENV XDG_CONFIG_HOME=/tmp/.config \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.config/matplotlib


ENV PYTHONUNBUFFERED=0

ENV PYTHONPATH=/workspace
 
WORKDIR /workspace

ENTRYPOINT []
CMD []




FROM base AS deploy
 








# DEV target -----------------------------

FROM deploy AS dev
 
 




 



 



#configure my dev user
ARG USERNAME=cefect
ARG USER_UID=1000
ARG USER_GID=1000
ARG OLD_USERNAME=ubuntu


USER root



# rename existing user if present; otherwise create the dev user
RUN set -eux; \
    if id -u "${OLD_USERNAME}" >/dev/null 2>&1; then \
        usermod -l "${USERNAME}" "${OLD_USERNAME}"; \
        usermod -d "/home/${USERNAME}" -m "${USERNAME}"; \
        groupmod -n "${USERNAME}" "${OLD_USERNAME}" || true; \
        usermod -u "${USER_UID}" "${USERNAME}" || true; \
        groupmod -g "${USER_GID}" "${USERNAME}" || true; \
    else \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
        useradd  --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi

# add to sudo + common dev tooling
RUN  --mount=type=cache,target=/var/cache/apt,sharing=locked \
    #--mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        git \
        git-lfs \
        openssh-client \
        #wget \
        rsync \
        procps \
        less \
       strace \
       gdb \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USERNAME} \
    && chmod 0440 /etc/sudoers.d/90-${USERNAME} \
    && rm -rf /var/lib/apt/lists/*

# Give the dev user ownership of places they'll modify (including conda dirs)
RUN set -eux; \
    install -d -o ${USER_UID} -g ${USER_GID} -m 0755 /workspace; \
    install -d -o ${USER_UID} -g ${USER_GID} -m 1777 /tmp /var/tmp; \
    mkdir -p /home/${USERNAME}; \
    chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}; 








#setup git: ship in a global gitconfig
COPY container/etc-gitconfig /etc/gitconfig
RUN chmod 644 /etc/gitconfig




# Starship prompt (x86_64) — tiny one-liner layer
RUN set -eux; \
    wget -qO /tmp/starship.tgz https://github.com/starship/starship/releases/latest/download/starship-x86_64-unknown-linux-gnu.tar.gz; \
    tar -xzf /tmp/starship.tgz -C /usr/local/bin starship; \
    echo 'eval "$(starship init bash)"' >> /etc/bash.bashrc; \
    rm -rf /tmp/starship.tgz





# setup analysis conda environment
COPY container/environment.yml /tmp/environment.yml
COPY container/environment_analysis.yml /tmp/environment_analysis.yml

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    mamba env create -yv --strict-channel-priority --name analysis --file /tmp/environment.yml && \
    mamba env update -y --name analysis --file /tmp/environment_analysis.yml && \
    rm -f /tmp/environment.yml /tmp/environment_analysis.yml

RUN conda env config vars set PYTHONPATH=/workspace --name analysis

# Install pipx system-wide into the base environment
# pipx docs recommend installing via pip and ensuring ~/.local/bin is on PATH:contentReference[oaicite:0]{index=0}
# RUN pip install --no-cache-dir pipx \
#  && python3 -m pipx ensurepath \
#  && ln -s /root/.local/bin/pipx /usr/local/bin/pipx

# RUN pipx install --pip-args="--no-cache-dir" conda-lock




#set default user
USER ${USER_UID}:${USER_GID}
  
# set default conda environment  
#NOTE: because the user already existed, the --target=deploy changes to their .bashrc need to be propagated
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${env_name}" >> ~/.bashrc

ENTRYPOINT ["/bin/bash", "-lc"]
