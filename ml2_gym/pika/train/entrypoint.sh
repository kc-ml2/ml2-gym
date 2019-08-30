#!/bin/bash

# Set X virtual framebuffer and VNC server
export DISPLAY=:0
Xvfb :0 -screen 0 640x480x24 &

# Run pika.exe
wine /app/assets/pika.exe &

# FIXME: change this to python command
/bin/bash
