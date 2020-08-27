# Buoywatch Detector
The Buoywatch detector is
## Installation 
Clone the repository
```bash
git clone https://github.com/fxs1l/Buoywatch.git
```
## Requirements
Python3.7 or later, OpenCV>=4.1.0, numpy>=1.16, Pytorch>=1.6, torchvision>=0.7.0, astral>=2.2

The object detection system relies on the Pytorch framework. SEArious has built pip wheels for armv7l architecture of the Raspberry Pi Model 3. The wheels can be found [here](https://drive.google.com/drive/folders/1bOt7IZvQqZWHa5XknjHfDmiuiRYoWuIE?usp=sharing). To install:

```bash
pip3 install torch-1.7.0a0+1f0cfba-cp37-cp37m-linux_armv7l.whl
pip3 install torchvision-0.8.0a0+fc69c22-cp37-cp37m-linux_armv7l.whl
```
Raspberry Pi camera saves videos in .h264 format. However, this is not a supported file format. Installing the the GPAC package allows for easy conversion to .mp4.
```bash
sudo apt install gpac
```

### Optional 
An optional feature of the Buoywatch detector automatically sends the inferenced images through Bluetooth OBEX Push service if any smartphones are nearby. To enable the feature, additional bluetooth packages are required.

```bash
sudo apt-get install bluetooth libbluetooth-dev
pip3 install pybluez
pip3 install PyOBEX
```
## Usage
The bash script ``run`` runs ``detect_boat.py`` and ``detect_light.py`` simultaneously. Make``run`` executable and run.
```bash 
chmod +x run
./run
```
### Enable on boot
Move ``run`` to ``etc/init.d/`` folder

