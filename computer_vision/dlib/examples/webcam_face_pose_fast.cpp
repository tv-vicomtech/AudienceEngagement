// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "render_face.hpp"

using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 2
#define OPENCV_FACE_RENDER

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        double fps = 30.0; // Just a place holder. Actual value calculated after 100 frames.
        cv::Mat im;
        
        // Get first frame and allocate memory.
        cap >> im;
        cv::Mat im_small, im_display;
        cv::resize(im, im_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
        cv::resize(im, im_display, cv::Size(), 0.5, 0.5);
        
        cv::Size size = im.size();

        
        
#ifndef OPENCV_FACE_RENDER 
        image_window win;
#endif

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        int count = 0;
        std::vector<rectangle> faces;
        // Grab and process frames until the main window is closed by the user.
        double t = (double)cv::getTickCount();
#ifdef OPENCV_FACE_RENDER
        while(1)
#else
        while(!win.is_closed())
#endif
        {
            
            if ( count == 0 )
                t = cv::getTickCount();
            // Grab a frame
            cap >> im;
            
            // Resize image for face detection
            cv::resize(im, im_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
            
            // Change to dlib's image format. No memory is copied.
            cv_image<bgr_pixel> cimg_small(im_small);
            cv_image<bgr_pixel> cimg(im);
            

            // Detect faces 
            if ( count % SKIP_FRAMES == 0 )
            {
                faces = detector(cimg_small);
            }
            

            
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
            {
                rectangle r(
                            (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
                            );
                full_object_detection shape = pose_model(cimg, r);
                shapes.push_back(shape);
#ifdef OPENCV_FACE_RENDER
                    render_face(im, shape);
#endif
            }
            
            cv::putText(im, cv::format("fps %.2f",fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            
            
            // Display it all on the screen
#ifdef OPENCV_FACE_RENDER
            
                // Resize image for display
                cv::resize(im, im_display, cv::Size(), 0.5, 0.5);
                cv::imshow("Fast Facial Landmark Detector", im_display);
            
                // WaitKey slows down the runtime quite a lot
                // So check every 15 frames
            
                if ( count % 15 == 0)
                {
                    int k = cv::waitKey(1);
                    // Quit if 'q' or ESC is pressed
                    if ( k == 'q' || k == 27)
                    {
                        return 0;
                    }
                }
            
 
#else
 
                win.clear_overlay();
                win.set_image(cimg);
                win.add_overlay(render_face_detections(shapes));
#endif
            
            count++;
            
            if ( count == 100)
            {
                t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
                fps = 100.0/t;
                count = 0;
            }
            

            
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

