ScreenReality
=============

###### *Project for the "IFT6145 - Vision 3D" class at University of Montréal.*
Creates an Augmented Reality through your computer screen, after detecting your eyes position.

### Prerequisites
* OpenCV 2.4.10 : `sudo apt-get install libopencv-dev`
* OpenGL & freeGLUT : `sudo apt-get install libxi-dev libxmu-dev freeglut3-dev`

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
* **I** : *inverts (flips upside-down) the camera image*
* **C** : *camera display ON/OFF*
* **D** : *detection information ON/OFF*
* **+/-** : *Adapts camera display size*
* **M** : *changes OpenGL PolygonMode between LINE and FILL*
* **P** : *changes OpenGL ProjectionMode between Off-Axis and Regular*
* **B** : *bounding box display ON/OFF*
