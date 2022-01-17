ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git vim wget sshfs ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMCV
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose

ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install -r requirements/mminstall.txt
RUN cd / && rm -rf /mmpose

#RUN git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
#RUN sh ~/.vim_runtime/install_awesome_vimrc.sh
#
#RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
#    -t robbyrussell \
#    -p git \
#    -p https://github.com/zsh-users/zsh-autosuggestions \
#    -p https://github.com/zsh-users/zsh-completions \
#    -p https://github.com/zsh-users/zsh-syntax-highlighting
RUN apt-get update && apt-get install -y zsh ctags
RUn chsh -s /bin/zsh
RUN sh -c "$(wget https://raw.githubusercontent.com/cavalleria/dotfiles/main/install.sh -O -)"
