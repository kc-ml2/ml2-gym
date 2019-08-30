FROM ubuntu:16.04
# FIXME: needs rearrangements

# Add 32-bit architecture and install packages
RUN dpkg --add-architecture i386
RUN export DEBIAN_FRONTEND="noninteractive" \
    && apt-get update \
    && apt-get install -y --install-recommends \
        bzip2 \
        curl \
        sudo \
        winbind \
        wine \
        xvfb \ 
    && rm -rf /var/lib/apt/lists/*

# Wine configurations
ENV WINEARCH win32
ADD asound.conf /etc/asound.conf

# Create a working directory
VOLUME /app
WORKDIR /app

# Create user with root permissions
RUN adduser --disabled-password --gecos '' user \
    && adduser user sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda3(4.5.11)
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

ADD entrypoint.sh /usr/bin/entrypoint
ENTRYPOINT /usr/bin/entrypoint
