FROM ubuntu:16.04
# modified from kxes/desktop

ARG xrdp_source=https://github.com/neutrinolabs/xrdp/releases/download/v0.9.3.1/xrdp-0.9.3.1.tar.gz
ARG xorgxrdp_source=https://github.com/neutrinolabs/xorgxrdp/releases/download/v0.2.3/xorgxrdp-0.2.3.tar.gz

# Add 32-bit architecture and install packages
RUN dpkg --add-architecture i386 \
    && export DEBIAN_FRONTEND="noninteractive" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        curl \
        dbus-x11 \
        file \
        gtk2-engines-pixbuf \
        libpam0g-dev \
        libssl-dev \
        libxfixes-dev \
        libxfont1-dev \
        libxrandr-dev \
        nasm \
        pkg-config \
        software-properties-common \
        sudo \
        vnc4server \
        wget \
        xfce4 \
        xorg \
        xserver-xorg \
        xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build and install Xrdp from source
RUN cd /tmp \
    && wget --no-check-certificate $xrdp_source \
    && tar -xf xrdp-*.tar.gz -C /tmp/ \
    && cd /tmp/xrdp-* \
    && ./configure \
    && make \
    && make install \
    && cd /tmp \
    && rm -rf xrdp-* \
    && wget --no-check-certificate $xorgxrdp_source \
    && tar -xf xorgxrdp-*.tar.gz -C /tmp/ \
    && cd /tmp/xorgxrdp-* \
    && ./configure \
    && make \
    && make install \
    && cd /tmp \
    && rm -rf xorgxrdp-* \
    && apt-get remove -y \
        wget \
        build-essential \
        libssl-dev \
        libpam0g-dev \
        libxrandr-dev \
        nasm \
        xserver-xorg-dev \
        libxfont1-dev \
        pkg-config \
        file \
        libxfixes-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Numix theme and icons
RUN add-apt-repository ppa:numix/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        numix-icon-theme \
        numix-icon-theme-circle \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add customised files from kxes/desktop
COPY ubuntu-files/lightdm-gtk-greeter.conf /etc/lightdm/lightdm-gtk-greeter.conf
COPY ubuntu-files/Adwaita-Xfce /usr/share/themes/Adwaita-Xfce
COPY ubuntu-files/xfce-perchannel-xml /etc/xdg/xfce4/xfconf/xfce-perchannel-xml
RUN ln -s /usr/share/icons/Numix-Circle /usr/share/icons/KXicons \
    && mkdir -p /usr/share/backgrounds
COPY ubuntu-files/background-default.png /usr/share/backgrounds/background-default.png

# Install Wine
RUN apt-get update \
    && apt-get install -y --install-recommends wine \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY asound.conf /etc/asound.conf

# Create working directory
VOLUME /app
WORKDIR /app

# Add default user
RUN useradd --create-home wontae \
    && adduser wontae sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && echo "wontae:chu" | chpasswd
USER wontae
ENV HOME=/home/wontae
RUN chmod 777 /home/wontae

# Install Miniconda3(4.5.11)
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT /entrypoint.sh
