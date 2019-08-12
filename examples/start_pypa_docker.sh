#! /bin/bash

sudo docker run -it --rm -v /home/rfq-dip-student/jared/:/home/pypa -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY pypa