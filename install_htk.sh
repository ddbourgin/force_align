#!/usr/bin/env bash

# Get HTK
read -p "HTK username: " htkuser
wget http://htk.eng.cam.ac.uk/ftp/software/HTK-3.4.tar.gz --user=$htkuser --ask-password

# Install HTK
tar xvfz HTK-3.4.tar.gz
cd htk
./configure --without-x --disable-hslab
make all && sudo make install
