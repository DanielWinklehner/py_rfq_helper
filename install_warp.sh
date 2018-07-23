sudo apt-get install --assume-yes python-pip python3-pip
sudo apt-get install --assume-yes gfortran
sudo apt-get install --assume-yes libx11-dev
sudo apt-get install --assume-yes git
sudo python -m pip install Forthon
sudo python3 -m pip install Forthon
sudo python -m pip install numpy
sudo python3 -m pip install numpy
sudo python -m pip install scipy
sudo python3 -m pip install scipy
sudo python -m pip install matplotlib
sudo python3 -m pip install matplotlib
sudo apt-get install python-tk
sudo apt-get install python3-tk
mkdir ~/src
cd ~/src
git clone https://bitbucket.org/dpgrote/pygist.git
cd ~/src/pygist
python setup.py config
sudo python setup.py install 
python3 setup.py config
sudo python3 setup.py install
cd ~/src
wget https://bitbucket.org/berkeleylab/warp/downloads/Warp_Release_4.5.tgz
tar -zxf Warp_Release_4.5.tgz warp
cd ~/src/warp
git pull
cd ~/src/warp/pywarp90
sudo make install
sudo make install3
cd ~/src/warp/warp_test
python runalltests.py
python3 runalltests.py
