#!/bin/bash

# Run Xrdp and session manager
sudo dbus-daemon --system
sudo /bin/sh /usr/share/xrdp/socksetup
sudo xrdp-sesman -ns &
sudo xrdp -ns &
