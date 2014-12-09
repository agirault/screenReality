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
const float sceneRatio = 1.5;
const bool bDisplayDetection = true;

const int minFaceSize = 100; // in pixel. The smaller it is, the further away you can go
const float cvCamViewAngleXDeg = 48.55;
const float cvCamViewAngleYDeg = 40.37;
const float eyesGap = 6.5; //cm


/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "/home/alexis/Support/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
cv::VideoCapture *capture = NULL;
cv::Mat frame;

float glCamX;
float glCamY;
float glCamZ;

/** Functions */
void redisplay();
cv::Mat detectEyes(cv::Mat image);

void onReshape( int w, int h );
void onMouse( int button, int state, int x, int y );
void onKeyboard( unsigned char key, int x, int y );
void onIdle();
void drawAxes(float length);

/**
 * @function main
 */
int main( int argc, char **argv )
{
    // OPENCV INIT
    //-- Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    { printf("-- (!) ERROR loading 'haarcascade_frontalface_alt.xml'\nPlease edit face_cascade_name in source code.\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    { printf("-- (!) ERROR loading 'haarcascade_eye_tree_eyeglasses.xml'\nPlease edit eyes_cascade_name in source code.\n"); return -1; };

    //-- Create Windows
    cv::namedWindow(window_face_detection);
    cv::moveWindow(window_face_detection, 400, 100);


    // VIDEO CAPTURE
    //-- start video capture from camera
    capture = new cv::VideoCapture(0);

    //-- check that video is working
    if ( capture == NULL || !capture->isOpened() ) {
        fprintf( stderr, "Could not start video capture\n" );
        return 1;
    }

    // WINDOW DIMENSIONS
    int width = (int) capture->get( CV_CAP_PROP_FRAME_WIDTH );
    int height = (int) capture->get( CV_CAP_PROP_FRAME_HEIGHT );

    // GLUT INIT
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowPosition( 20, 20 );
    glutInitWindowSize( width*sceneRatio, height*sceneRatio );
    glutCreateWindow( "Test OpenGL" );

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
    glClear( GL_COLOR_BUFFER_BIT );

    /** OPENCV */
    //-- flip frame image
    cv::Mat tempimage;
    cv::flip(frame, tempimage, 0);
    //-- detect eyes
    tempimage = detectEyes(tempimage);


    /** OPENGL */
    // CAMERA PARAMETERS
    //-- set projection matrix using intrinsic camera params
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //-- gluPerspective is arbitrarily set, you will have to determine these values based
    gluPerspective(60, tempimage.size().width*1.0/tempimage.size().height, 1, 20);
    //-- you will have to set modelview matrix using extrinsic camera params
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);

    // 3D SCENE
    glPushMatrix();
    drawAxes(1.0);
    glutSolidTeapot(1.0);
    glTranslatef(0.0f, 0.0f, 2.0f);
    glPopMatrix();

    //-- display camera frame
    cv::flip(tempimage, tempimage, 0);
    cv::resize(tempimage, tempimage, cv::Size(camRatio*tempimage.size().width, camRatio*tempimage.size().height), 0, 0, cv::INTER_CUBIC);
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

        // COORDINATES
        //-- z
        int imgEyesGap = rightEyePt.x-leftEyePt.x;
        glCamZ = (550.0*eyesGap)/imgEyesGap; // doesn't look right
        //-- x
        float ratioX = (float)eyeCenterPt.x/image.size().width;
        if(ratioX <= 0.5){
            float phi1 = cvCamViewAngleXDeg * ratioX;
            float phi2 = (cvCamViewAngleXDeg/2.0)-phi1;
            glCamX = -glCamZ*tan(phi2*M_PI/180.0);
        }else if(ratioX > 0.5){
            float phi1 = cvCamViewAngleXDeg * ratioX;
            float phi2 = phi1-(cvCamViewAngleXDeg/2.0);
            glCamX = glCamZ*tan(phi2*M_PI/180.0);
        }
        //-- y
        float ratioY = (float)eyeCenterPt.y/image.size().height;
        if(ratioY <= 0.5){
            float phi1 = cvCamViewAngleYDeg * ratioY;
            float phi2 = (cvCamViewAngleYDeg/2.0)-phi1;
            glCamY = glCamZ*tan(phi2*M_PI/180.0);
        }else if(ratioY > 0.5){
            float phi1 = cvCamViewAngleYDeg * ratioY;
            float phi2 = phi1-(cvCamViewAngleYDeg/2.0);
            glCamY = -glCamZ*tan(phi2*M_PI/180.0);
        }
        //-- cout
        std::cout<<"(x,y,z) = ("<<glCamX<<","<<glCamY<<","<<glCamZ<<")"<<std::endl;

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

/**
 * @function drawAxes
 * A useful function for displaying your coordinate system
 */
void drawAxes(float length)
{
  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT) ;

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) ;
  glDisable(GL_LIGHTING) ;

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

  glPopAttrib() ;
}
