#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <GL/gl.h>
#include <GL/freeglut.h>

#include <iostream>
#include <stdio.h>

/** Constants */
const int minFaceSize = 80; // in pixel. The smaller it is, the further away you can go
const float f = 500; //804.71
const float eyesGap = 6.5; //cm
const float pixelNbrPerCm = 40.0;
const float far = 200;
const float near = 0.5;

/** Global variables */
//-- capture opencv
cv::String face_cascade_name = "../../haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
cv::VideoCapture *capture = NULL;
cv::Mat frame;

//-- display
bool bFullScreen = true;            //- press 'f' to change
bool bDisplayCam = true;            //- press 'c' to change
bool bDisplayDetection = true;      //- press 'd' to change
bool bPolygonMode = false;          //- press 'm' to change
bool bProjectionMode = true;        //- press 'p' to change
float camRatio = 0.3;               //- press '+/-' to change
float angleRotY = 0.0;              //- press 'LEFT/RIGHT' to change
float angleRotX = 0.0;              //- press 'UP/DOWN' to change

//-- dimensions
int windowWidth;
int windowHeight;
int camWidth;
int camHeight;

//-- opengl camera
GLdouble glCamX;
GLdouble glCamY;
GLdouble glCamZ;

/** Functions */
void redisplay();
cv::Mat detectEyes(cv::Mat image);

void setGlCamera();
void draw3dScene();
void displayCam(cv::Mat camImage);

void drawScreenFrame();
void drawCube(float x, float y, float z, float l, float angle, float ax, float ay, float az );
void drawAxes(float length);

void onReshape( int w, int h );
void onMouse( int button, int state, int x, int y );
void onKeyboard( unsigned char key, int x, int y );
void onSpecialKey(int key, int x, int y);
void onIdle();


/**
 * @function main
 */
int main( int argc, char **argv )
{
    // OPENCV INIT
    //-- Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    { printf("-- (!) ERROR loading 'haarcascade_frontalface_alt.xml'\nPlease edit face_cascade_name in source code.\n"); return -1; };

    // VIDEO CAPTURE
    //-- start video capture from camera
    capture = new cv::VideoCapture(0);

    //-- check that video is working
    if ( capture == NULL || !capture->isOpened() ) {
        fprintf( stderr, "Could not start video capture\n" );
        return 1;
    }

    // CAMERA IMAGE DIMENSIONS
    camWidth = (int) capture->get( CV_CAP_PROP_FRAME_WIDTH );
    camHeight = (int) capture->get( CV_CAP_PROP_FRAME_HEIGHT );

    // GLUT INIT
    glutInit( &argc, argv );
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    if(!bFullScreen){
        windowWidth = camWidth*1.5;
        windowHeight = camHeight*1.5;
        glutInitWindowPosition( 200, 80 );
        glutInitWindowSize( windowWidth, windowHeight );
    }
    glutCreateWindow( "ScreenReality - Vision 3D" );

    // RENDERING INIT
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0); //Enable light #0
    glEnable(GL_LIGHT1); //Enable light #1
    glEnable(GL_NORMALIZE); //Automatically normalize normals

    // SCREEN DIMENSIONS
    if(bFullScreen)
    {
        glutFullScreen();
        windowWidth = glutGet(GLUT_SCREEN_WIDTH);
        windowHeight = glutGet(GLUT_SCREEN_HEIGHT);
        glViewport( 0, 0, windowWidth, windowHeight );
    }

    // GUI CALLBACK FUNCTIONS
    glutDisplayFunc( redisplay );
    glutReshapeFunc( onReshape );
    glutMouseFunc( onMouse );
    glutKeyboardFunc( onKeyboard );
    glutSpecialFunc( onSpecialKey );
    glutIdleFunc( onIdle );

    // GUI LOOP
    glutMainLoop();

    return 0;
}

/**
 * @function redisplay
 * (Called at each openGL step)
 * - Processes the webcam frame to detect the eyes with OpenCV,
 * - Creates a 3D scene with OpenGL,
 * - Render the scene and the webcam image.
 */
void redisplay()
{
    if(frame.empty()) {return;}
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // OPENCV
    //-- flip frame image
    cv::Mat tempimage;
    cv::flip(frame, tempimage, 0);
    //-- detect eyes
    tempimage = detectEyes(tempimage);

    // OPENGL
    //-- scene
    setGlCamera();
    draw3dScene();
    //-- cam
    if(bDisplayCam) displayCam(tempimage);

    // RENDER
    //-- display on screen
    glutSwapBuffers();
    //-- post the next redisplay
    glutPostRedisplay();
}

/**
 * @function detectEyes
 * - Uses OpenCV to detect face
 * - Interpolate eyes position on image
 * - Computes eyes position in space
 * - Add some display for the detection
 */
cv::Mat detectEyes(cv::Mat image)
{
    // INIT
    std::vector<cv::Rect> faces;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    cv::equalizeHist( image_gray, image_gray );

    // DETECT FACE
    //-- Find bigger face (opencv documentation)
    face_cascade.detectMultiScale( image_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(minFaceSize, minFaceSize) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
        // DETECT EYES
        //-- points in pixel
        cv::Point leftEyePt( faces[i].x + faces[i].width*0.30, faces[i].y + faces[i].height*0.37 );
        cv::Point rightEyePt( faces[i].x + faces[i].width*0.70, faces[i].y + faces[i].height*0.37 );
        cv::Point eyeCenterPt( faces[i].x + faces[i].width*0.5, leftEyePt.y );

        //-- normalize with webcam internal parameters
        GLdouble normRightEye = (rightEyePt.x - camWidth/2)/f;
        GLdouble normLeftEye = (leftEyePt.x - camWidth/2)/f;
        GLdouble normCenterX = (eyeCenterPt.x - camWidth/2)/f;
        GLdouble normCenterY = (eyeCenterPt.y - camHeight/2)/f;

        //-- get space coordinates
        glCamZ = eyesGap/(normRightEye-normLeftEye);
        glCamX = normCenterX*glCamZ;
        glCamY = -normCenterY*glCamZ;

        // DISPLAY
        if(bDisplayCam && bDisplayDetection)
        {
            //-- face rectangle
            cv::rectangle(image, faces[i], 1234);

            //-- face lines
            cv::Point leftPt( faces[i].x, faces[i].y + faces[i].height*0.37 );
            cv::Point rightPt( faces[i].x + faces[i].width, faces[i].y + faces[i].height*0.37 );
            cv::Point topPt( faces[i].x + faces[i].width*0.5, faces[i].y);
            cv::Point bottomPt( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height);
            cv::line(image, leftPt, rightPt, cv::Scalar( 0, 0, 0 ), 1, 1, 0);
            cv::line(image, topPt, bottomPt, cv::Scalar( 0, 0, 0 ), 1, 1, 0);

            //-- eyes circles
            cv::circle(image, rightEyePt, 0.06*faces[i].width, cv::Scalar( 255, 255, 255 ), 1, 8, 0);
            cv::circle(image, leftEyePt, 0.06*faces[i].width, cv::Scalar( 255, 255, 255 ), 1, 8, 0);

            //-- eyes line & center
            cv::line(image, leftEyePt, rightEyePt, cv::Scalar( 0, 0, 255 ), 1, 1, 0);
            cv::circle(image, eyeCenterPt, 2, cv::Scalar( 0, 0, 255 ), 3, 1, 0);
        }
    }
    return image;
}

/**
 * @function setGlCamera
 * Set OpenGL camera parameters.
 * The off-axis projection is what gives
 * the feeling of augmented reality.
 */
void setGlCamera()
{
    if(bProjectionMode)
    {
        /* SKEWED FRUSTRUM / OFF-AXIS PROJECTION
        ** My implementation is based on the following paper:
        ** Name:   Generalized Perspective Projection
        ** Author: Robert Kooima
        ** Date:   August 2008, revised June 2009
        */

        float cx = (float)windowWidth/pixelNbrPerCm;
        float cy = (float)windowHeight/pixelNbrPerCm;

        //-- space corners coordinates
        float pa[3]={-cx,-cy,0};
        float pb[3]={cx,-cy,0};
        float pc[3]={-cx,cy,0};
        float pe[3]={glCamX,glCamY,glCamZ};
        //-- space points
        cv::Vec3f Pa(pa);
        cv::Vec3f Pb(pb);
        cv::Vec3f Pc(pc);
        cv::Vec3f Pe(pe);
        //-- Compute an orthonormal basis for the screen.
        cv::Vec3f Vr = Pb-Pa;
        Vr /= cv::norm(Vr);
        cv::Vec3f Vu = Pc-Pa;
        Vu /= cv::norm(Vu);
        cv::Vec3f Vn = Vr.cross(Vu);
        Vn /= cv::norm(Vn);
        //-- Compute the screen corner vectors.
        cv::Vec3f Va = Pa-Pe;
        cv::Vec3f Vb = Pb-Pe;
        cv::Vec3f Vc = Pc-Pe;
        //-- Find the distance from the eye to screen plane.
        float d = -Va.dot(Vn);
        //-- Find the extent of the perpendicular projection.
        float l = Va.dot(Vr) * near / d;
        float r = Vr.dot(Vb) * near / d;
        float b = Vu.dot(Va) * near / d;
        float t = Vu.dot(Vc) * near / d;
        //-- Load the perpendicular projection.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glFrustum(l, r, b, t, near, far);
        //-- Rotate the projection to be non-perpendicular.
        float M[16];
        memset(M, 0, 16 * sizeof (float));
        M[0] = Vr[0]; M[1] = Vu[0]; M[2] = Vn[0];
        M[4] = Vr[1]; M[5] = Vu[1]; M[6] = Vn[1];
        M[8] = Vr[2]; M[9] = Vu[2]; M[10] = Vn[2];
        M[15] = 1.0f;
        glMultMatrixf(M);
        //-- Move the apex of the frustum to the origin.
        glTranslatef(-pe[0], -pe[1], -pe[2]);
        //-- Reset modelview matrix
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
    else
    {
        //-- intrinsic camera params
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60, (float)windowWidth/(float)windowHeight, 1, 250);
        //-- extrinsic camera params
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(glCamX, glCamY, glCamZ, 0, 0, 0, 0, 1, 0);
    }
}

/**
 * @function draw3dScene
 * Draws OpenGL 3D scene
 */
void draw3dScene()
{
    // DISPLAY MODE
    if(bPolygonMode){
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        drawAxes(10.0);
    }else{
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // SCREEN BORDERS
    if(!bProjectionMode){
        glColor3f(1.0f, 0.0f, 0.0f);
        drawScreenFrame();
    }

    // LIGHTING
    glEnable(GL_LIGHTING); //Enable lighting
    //-- Add ambient light
    GLfloat ambientColor[] = {0.2f, 0.2f, 0.2f, 1.0f}; //Color (0.2, 0.2, 0.2)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientColor);
    //-- Add positioned light
    GLfloat lightColor0[] = {0.5f, 0.5f, 0.5f, 1.0f}; //Color (0.5, 0.5, 0.5)
    GLfloat lightPos0[] = {4.0f, 0.0f, 8.0f, 1.0f}; //Positioned at (4, 0, 8)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor0);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
    //-- Add directed light
    GLfloat lightColor1[] = {0.5f, 0.2f, 0.2f, 1.0f}; //Color (0.5, 0.2, 0.2)
    GLfloat lightPos1[] = {-1.0f, 0.5f, 0.5f, 0.0f}; //Positioned at (-1, 0.5, 0.5)
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor1);
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);

    // MOVE SCENE
    //glTranslatef(x, y, z);
    glRotatef(angleRotX, 1.0, 0.0, 0.0);
    glRotatef(angleRotY, 0.0, 1.0, 0.0);

    // GEOMETRY
    //-- TeaPot
    glColor3f(1.0f, 1.0f, 1.0f);
    glutSolidTeapot(5.0);
    //-- Cube 1
    //glColor3f(1.0f, 1.0f, 0.0f);
    //drawCube(0.0, 0.0, 0.0, 6.0, 30.0, 0.0, 1.0, 0.0 );
    //-- Cube 2
    glColor3f(1.0f, 0.0f, 1.0f);
    drawCube(-20.0, 0.0, -40.0, 3.0, 70.0, 0.0, 1.0, 0.0 );
    //-- Cube 3
    glColor3f(0.0f, 1.0f, 1.0f);
    drawCube(5.0, 5.0, 10.0, 3.0, 10.0, 0.0, 1.0, 0.0 );

    glDisable(GL_LIGHTING); //Disable lighting

}

/**
 * @function displayCam
 * Draws the webcam image in window + detection info
 */
void displayCam(cv::Mat camImage)
{
    //-- Save matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, windowWidth, 0.0, windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    //-- Display Coordinates
    if(bDisplayDetection)
    {
        //-- Coord text
        std::stringstream sstm;
        sstm << "(x,y,z) = (" << (int)glCamX << "," << (int)glCamY << "," << (int)glCamZ << ")";
        std::string s = sstm.str();
        //std::cout<<s<<std::endl;

        //-- Display text
        glColor3f(1.0, 1.0, 1.0);
        glRasterPos2i(10,  windowHeight-(camRatio*camImage.size().height)-20);
        void * font = GLUT_BITMAP_9_BY_15;
        for (std::string::iterator i = s.begin(); i != s.end(); ++i)
        {
          char c = *i;
          glutBitmapCharacter(font, c);
        }
    }

    //-- Display image
    glRasterPos2i(0, windowHeight-(camRatio*camImage.size().height));
    cv::flip(camImage, camImage, 0);
    cv::resize(camImage, camImage, cv::Size(camRatio*camWidth, camRatio*camHeight), 0, 0, cv::INTER_CUBIC);
    glDrawPixels( camImage.size().width, camImage.size().height, GL_BGR, GL_UNSIGNED_BYTE, camImage.ptr() );

    //-- Load matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

/**
 * @function drawScreenFrame
 * Draws lines between the corners of what we consider to be the screen
 * (We are now attempting to project that image on the screen with an homography)
 */
void drawScreenFrame()
{
    float cx = (float)windowWidth/pixelNbrPerCm;
    float cy = (float)windowHeight/pixelNbrPerCm;

    glBegin(GL_LINE_LOOP);
        glVertex3f(cx, cy, 0);
        glVertex3f(cx, -cy, 0);
        glVertex3f(-cx, -cy, 0);
        glVertex3f(-cx, cy, 0);
    glEnd();
}

/**
 * @function drawCube
 * Draws a cube object
 */
void drawCube(float x, float y, float z, float l, float angle, float ax, float ay, float az )
{
    glTranslatef(x, y, z);
    glRotatef(angle, ax, ay, az);

    glBegin(GL_QUADS);

        //Front
        glNormal3f(0.0, 0.0, 1.0);
        glVertex3f(-l, -l, l);
        glVertex3f(l, -l, l);
        glVertex3f(l, l, l);
        glVertex3f(-l, l, l);

        //Right
        glNormal3f(1.0, 0.0, 0.0);
        glVertex3f(l, -l, -l);
        glVertex3f(l, l, -l);
        glVertex3f(l, l, l);
        glVertex3f(l, -l, l);

        //Back
        glNormal3f(0.0, 0.0, -1.0);
        glVertex3f(-l, -l, -l);
        glVertex3f(-l, l, -l);
        glVertex3f(l, l, -l);
        glVertex3f(l, -l, -l);

        //Left
        glNormal3f(-1.0, 0.0, 0.0);
        glVertex3f(-l, -l, -l);
        glVertex3f(-l, -l, l);
        glVertex3f(-l, l, l);
        glVertex3f(-l, l, -l);

        //Top
        glNormal3f(0.0, 1.0, 0.0);
        glVertex3f(-l, l, -l);
        glVertex3f(-l, l, l);
        glVertex3f(l, l, l);
        glVertex3f(l, l, -l);

        //Bottom
        glNormal3f(0.0, -1.0, 0.0);
        glVertex3f(-l, -l, -l);
        glVertex3f(-l, -l, l);
        glVertex3f(l, -l, l);
        glVertex3f(l, -l, -l);

    glEnd();
    glRotatef(-angle, ax, ay, az);
    glTranslatef(-x, -y, -z);
}

/**
 * @function drawAxes
 ** Display the coordinate system
 */
void drawAxes(float length)
{

  glBegin(GL_LINES) ;
      glColor3f(1,0,0) ;
      glVertex3f(0,0,0) ;
      glVertex3f(length,0,0);

      glColor3f(0,1,0) ;
      glVertex3f(0,0,0) ;
      glVertex3f(0,length,0);

      glColor3f(0,0,1) ;
      glVertex3f(0,0,0) ;
      glVertex3f(0,0,length);
  glEnd() ;

}

/**
 * @function onReshape;
 ** Adapts the viewport to the window size when the window is reshaped
 */
void onReshape( int w, int h )
{
    windowWidth = w;
    windowHeight = h;
    glViewport( 0, 0, windowWidth, windowHeight );
}

/**
 * @function onMouse
 */
void onMouse( int button, int state, int x, int y )
{
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_UP )
    {

    }
}

/**
 * @function onKeyboard
 */
void onKeyboard( unsigned char key, int x, int y )
{
    if( bFullScreen && ((key == 'f' || key == 'F') || ((int)key == 27)))
    {
        glutReshapeWindow(camWidth*1.5, camHeight*1.5);
        glutPositionWindow( 200, 80);
        bFullScreen = false;
    }
    else if(!bFullScreen && (key == 'f' || key == 'F'))
    {
        glutFullScreen();
        bFullScreen = true;
    }
    else switch ( key )
    {
        // change cam display
        case 'c': bDisplayCam = !bDisplayCam; break;
        case 'C': bDisplayCam = !bDisplayCam; break;

        // change detection display
        case 'd': bDisplayDetection = !bDisplayDetection; break;
        case 'D': bDisplayDetection = !bDisplayDetection; break;

        // change cam ratio
        case '+': if(camRatio < 1.8) camRatio += 0.1 ; break;
        case '-': if(camRatio > 0.2) camRatio -= 0.1; break;

        // change axes display
        case 'm': bPolygonMode = !bPolygonMode; break;
        case 'M': bPolygonMode = !bPolygonMode; break;

        // change homography correction
        case 'p': bProjectionMode = !bProjectionMode; break;
        case 'P': bProjectionMode = !bProjectionMode; break;

        // quit app
        case 'q': exit(0); break;
        case 'Q': exit(0); break;

        default: break;
    }
}

void onSpecialKey( int key, int x, int y)
{
    switch (key)
    {
    case GLUT_KEY_LEFT:
        angleRotY -= 1.0;
        break;
    case GLUT_KEY_RIGHT:
        angleRotY += 1.0;
        break;
    case GLUT_KEY_DOWN:
        angleRotX += 1.0;
        break;
    case GLUT_KEY_UP:
        angleRotX -= 1.0;
        break;
    }
}
/**
 * @function onIdle
 * (Called at each openGL step)
 * Updates the 'frame' image from the captured video
 */
void onIdle()
{
    (*capture) >> frame;
}

