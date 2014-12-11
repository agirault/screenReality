ScreenReality
=============

###### *Project for the "IFT6145 - Vision 3D" class at University of Montréal.*
Creates an Augmented Reality through your computer screen, after detecting your eyes position.

### Prerequisites
* OpenCV : [Installation in Linux](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html)
* OpenGL & GLUT : `sudo apt-get install freeglut3-dev`

### Build
In a Linux Terminal: 
```
git clone https://github.com/agirault/screenReality.git
cd screenReality
mkdir build
cd build
cmake ..
make
./bin/screenReality
```
### Shortcuts
* **Q** : *exit application*
* **F** : *fullscreen ON/OFF*
* **C** : *camera display ON/OFF*
* **D** : *detection information ON/OFF*
* **+/-** : *Adapts camera display size*
* **M** : *changes OpenGL PolygonMode between LINE and FILL*
* **P** : *changes OpenGL ProjectionMode between Off-Axis and Regular*
