#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <GL/gl.h>
#include <GL/freeglut.h>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

/** Constants */
const float camRatio = 0.5;
const float windowRatio = 1.0;
const bool bDisplayDetection = true;
const bool bFullScreen = true;

const int minFaceSize = 80; // in pixel. The smaller it is, the further away you can go
const float cvCamViewAngleXDeg = 48.55;
const float cvCamViewAngleYDeg = 40.37;
const float cx = 362.9;
const float cy = 395.84;
const float f = 804.71;
const float eyesGap = 6.5; //cm


/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "/home/alexis/Support/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
cv::VideoCapture *capture = NULL;
cv::Mat frame;

int windowWidth;
int windowHeight;
int camWidth;
int camHeight;

float glCamX;
float glCamY;
float glCamZ;

/** Functions */
void redisplay();
cv::Mat detectEyes(cv::Mat image);

void setGlCamera();
void draw3dScene();
void drawCube();
void drawAxes(float length);

void onReshape( int w, int h );
void onMouse( int button, int state, int x, int y );
void onKeyboard( unsigned char key, int x, int y );
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
        windowWidth = camWidth*windowRatio;
        windowHeight = camHeight*windowRatio;
        glutInitWindowPosition( 20, 20 );
        glutInitWindowSize( windowWidth, windowHeight );
    }
    glutCreateWindow( "Test OpenGL" );

    // RENDERING INIT
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING); //Enable lighting
    glEnable(GL_LIGHT0); //Enable light #0
    glEnable(GL_LIGHT1); //Enable light #1
    glEnable(GL_NORMALIZE); //Automatically normalize normals
    glShadeModel(GL_SMOOTH); //Enable smooth shading

    // SCREEN DIMENSIONS
    windowWidth = glutGet(GLUT_SCREEN_WIDTH);
    windowHeight = glutGet(GLUT_SCREEN_HEIGHT);
    if(bFullScreen)
    {
        glutFullScreen();
        glViewport( 0, 0, windowWidth, windowHeight );
    }

    // GUI CALLBACK FUNCTIONS
    glutDisplayFunc( redisplay );
    glutReshapeFunc( onReshape );
    glutMouseFunc( onMouse );
    glutKeyboardFunc( onKeyboard );
    glutIdleFunc( onIdle );

    // GUI LOOP
    glutMainLoop();

    return 0;
}

/**
 * @function display
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
    setGlCamera();
    draw3dScene();

    //-- display camera frame
    cv::flip(tempimage, tempimage, 0);
    cv::resize(tempimage, tempimage, cv::Size(camRatio*camWidth, camRatio*camHeight), 0, 0, cv::INTER_CUBIC);
    glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );

    // RENDER
    //-- display on screen
    glutSwapBuffers();
    //-- post the next redisplay
    glutPostRedisplay();
}

cv::Mat detectEyes(cv::Mat image) {

    // INIT
    std::vector<cv::Rect> faces;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    cv::equalizeHist( image_gray, image_gray );

    // FACE
    //-- Find bigger face (opencv documentation)
    face_cascade.detectMultiScale( image_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(minFaceSize, minFaceSize) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
        // EYES
        cv::Point leftEyePt( faces[i].x + faces[i].width*0.30, faces[i].y + faces[i].height*0.37 );
        cv::Point rightEyePt( faces[i].x + faces[i].width*0.70, faces[i].y + faces[i].height*0.37 );
        cv::Point eyeCenterPt( faces[i].x + faces[i].width*0.5, leftEyePt.y );

        float testX1 = (rightEyePt.x - cx)/f;
        float testX2 = (leftEyePt.x - cx)/f;
        float spaceZ = 0.5*eyesGap/(testX1-testX2);
        std::cout<<spaceZ<<std::endl;

        // COORDINATES
        //-- z
        int imgEyesGap = rightEyePt.x-leftEyePt.x;
        float tempsGLCamZ = (550.0*eyesGap)/imgEyesGap;
        glCamZ = (tempsGLCamZ+glCamZ)/2.0;
        //-- x
        float ratioX = (float)eyeCenterPt.x/camWidth;
        if(ratioX <= 0.5){
            float phi1 = cvCamViewAngleXDeg * ratioX;
            float phi2 = (cvCamViewAngleXDeg/2.0)-phi1;
            float tempGLCamX = -glCamZ*tan(phi2*M_PI/180.0);
            glCamX = (tempGLCamX+glCamX)/2.0;
        }else if(ratioX > 0.5){
            float phi1 = cvCamViewAngleXDeg * ratioX;
            float phi2 = phi1-(cvCamViewAngleXDeg/2.0);
            float tempGLCamX = glCamZ*tan(phi2*M_PI/180.0);
            glCamX = (tempGLCamX+glCamX)/2.0;
        }
        //-- y
        float ratioY = (float)eyeCenterPt.y/camHeight;
        if(ratioY <= 0.5){
            float phi1 = cvCamViewAngleYDeg * ratioY;
            float phi2 = (cvCamViewAngleYDeg/2.0)-phi1;
            float tempGLCamY = glCamZ*tan(phi2*M_PI/180.0);
            glCamY = (tempGLCamY+glCamY)/2.0;
        }else if(ratioY > 0.5){
            float phi1 = cvCamViewAngleYDeg * ratioY;
            float phi2 = phi1-(cvCamViewAngleYDeg/2.0);
            float tempGLCamY = -glCamZ*tan(phi2*M_PI/180.0);
            glCamY = (tempGLCamY+glCamY)/2.0;
        }
        //-- cout
        //std::cout<<"(x,y,z) = ("<<(int)glCamX<<","<<(int)glCamY<<","<<(int)glCamZ<<")"<<std::endl;

        //DISPLAY
        if(bDisplayDetection)
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

void setGlCamera()
{
    // CAMERA PARAMETERS
    //-- set projection matrix using intrinsic camera params
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //-- gluPerspective is arbitrarily set, you will have to determine these values based
    gluPerspective(60, windowWidth*1.0/windowHeight, 1, 20);
    //-- you will have to set modelview matrix using extrinsic camera params
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(glCamX/10.0, glCamY/10.0, glCamZ/10.0, 0, 0, 0, 0, 1, 0);
}

void draw3dScene()
{
    // LIGHTING
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
    //Coming from the direction (-1, 0.5, 0.5)
    GLfloat lightPos1[] = {-1.0f, 0.5f, 0.5f, 0.0f};
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor1);
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);

    // GEOMETRY
    //-- Pose and Color
    glTranslatef(0.0f, 0.0f, 0.0f);
    glRotatef(30.0f, 0.0f, 1.0f, 0.0f);
    glColor3f(1.0f, 1.0f, 0.0f);
    drawCube();

    //-- Pose and Color
    glTranslatef(-2.0f, 0.0f, -2.0f);
    glRotatef(70.0f, 0.0f, 1.0f, 0.0f);
    glColor3f(1.0f, 0.0f, 1.0f);
    drawCube();

    //-- Pose and Color
    glTranslatef(0.0f, 0.0f, 6.0f);
    glRotatef(10.0f, 0.0f, 1.0f, 0.0f);
    glColor3f(0.0f, 1.0f, 1.0f);
    drawCube();


}

void drawCube()
{
    glBegin(GL_QUADS);
    //Front
    glNormal3f(0.0f, 0.0f, 1.0f);
    //glNormal3f(-1.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, -1.0f, 1.0f);
    //glNormal3f(1.0f, 0.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, 1.0f);
    //glNormal3f(1.0f, 0.0f, 1.0f);
    glVertex3f(1.0f, 1.0f, 1.0f);
    //glNormal3f(-1.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, 1.0f);

    //Right
    glNormal3f(1.0f, 0.0f, 0.0f);
    //glNormal3f(1.0f, 0.0f, -1.0f);
    glVertex3f(1.0f, -1.0f, -1.0f);
    //glNormal3f(1.0f, 0.0f, -1.0f);
    glVertex3f(1.0f, 1.0f, -1.0f);
    //glNormal3f(1.0f, 0.0f, 1.0f);
    glVertex3f(1.0f, 1.0f, 1.0f);
    //glNormal3f(1.0f, 0.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, 1.0f);

    //Back
    glNormal3f(0.0f, 0.0f, -1.0f);
    //glNormal3f(-1.0f, 0.0f, -1.0f);
    glVertex3f(-1.0f, -1.0f, -1.0f);
    //glNormal3f(-1.0f, 0.0f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    //glNormal3f(1.0f, 0.0f, -1.0f);
    glVertex3f(1.0f, 1.0f, -1.0f);
    //glNormal3f(1.0f, 0.0f, -1.0f);
    glVertex3f(1.0f, -1.0f, -1.0f);

    //Left
    glNormal3f(-1.0f, 0.0f, 0.0f);
    //glNormal3f(-1.0f, 0.0f, -1.0f);
    glVertex3f(-1.0f, -1.0f, -1.0f);
    //glNormal3f(-1.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, -1.0f, 1.0f);
    //glNormal3f(-1.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, 1.0f);
    //glNormal3f(-1.0f, 0.0f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);

    glEnd();
}

/**
 * @function drawAxes
 * A useful function for displaying your coordinate system
 */
void drawAxes(float length)
{
  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT) ;

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) ;

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

  glPopMatrix() ;
}

/**
 * @function onReshape;
 */
void onReshape( int w, int h )
{
  //glViewport( 0, 0, w, h );
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
  switch ( key )
    {
    case 'q':
      // quit when q is pressed
      exit(0);
      break;

    default:
      break;
    }
}

/**
 * @function onIdle
 */
void onIdle()
{
    (*capture) >> frame;
}

