#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "stdio.h"
#include "string"
#include <stdlib.h>

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat K_Means(Mat Input, int K);

RNG rng(12345);

int main ()
{
    //Create a window with suitable name and size
    namedWindow( "RGB", 1 );
    //namedWindow( "Contour", 1 );

    //The Mat class of OpenCV library is used to store the values of an image. It represents an n-dimensional array and is used to store image data of grayscale or color images...
    Mat img, hsv, binary, binary1, binary2, binary3, binary4, binary5, imgToProcess;

    //Read the input image and show it
    //img = imread("stop_sign.jpg");
    //img = imread("priority_sign.png");
    img = imread("crosswalk_sign.png");
    //img = imread("parking_sign.png");
    //img = imread("proceed_straight_sign.png");
    //img = imread("one_way.png");
    //img = imread("danger_sign.png");
    //img = imread("no_parking_sign.png");
    //img = imread("no_priority_sign.png");
    //TODO img = imread("pedestrian_zone_sign.png");
    //img = imread("highway_entrance_sign.png");
    //img = imread("highway_exit_sign.png");
    //img = imread("black_spot_sign.png");

    imshow("RGB", img);
    waitKey(0);

    //Convert RGB image into HSV and show it
    cvtColor(img, hsv, CV_BGR2HSV);
    imshow("HSV", hsv);
    waitKey(0);


    //Get binary image(black and white)

    //Range for HSV values

    //Range for Blue
    inRange(hsv, Scalar(101, 150, 150), Scalar(140, 255, 255), binary);

    //Range for Red
    inRange(hsv, Scalar(0, 61, 38), Scalar(10, 255, 255), binary1);
    inRange(hsv, Scalar(170, 61, 38), Scalar(180, 255, 255), binary2);

    //Range for Green
    inRange(hsv, Scalar(46, 61, 38), Scalar(100, 255, 255), binary3);

    //Range for Yellow
    inRange(hsv, Scalar(21, 150, 150), Scalar(33, 255, 255), binary4);

    //Range for Black
    //inRange(hsv, Scalar(0, 0, 200), Scalar(180, 255, 255), binary5);

    //Adding every element from one array to another
    add(binary1, binary, imgToProcess, noArray(), 8);
    add(binary2, imgToProcess, imgToProcess, noArray(), 8);
    add(binary3, imgToProcess, imgToProcess, noArray(), 8);
    add(binary4, imgToProcess, imgToProcess, noArray(), 8);
    //add(binary5, imgToProcess, imgToProcess, noArray(), 8);

    imshow("Binaries", imgToProcess);
    waitKey(0);


    //Find contours from binary image
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(imgToProcess, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    //int max_ind_of_contour = getMaxAreaContourId(contours);

    Mat mask = Mat::zeros( imgToProcess.size(), CV_8UC3 );

    double MaxAreaContour = 50000;
    double MinAreaContour = 500;

    for( int i = 0; i< contours.size(); i++ ) {
        double AreaContour = contourArea(contours[i]);
        if (AreaContour < MaxAreaContour && AreaContour > MinAreaContour) {
            drawContours(mask, contours, i, Scalar(255, 255, 255), cv::FILLED);
        }
    }

    //drawContours(mask, contours, max_ind_of_contour, Scalar(255, 255, 255), cv::FILLED);

    imshow("Filled Mask", mask);
    waitKey(0);


    //Change background color

    for(int i=0; i<img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (mask.at<Vec3b>(i, j)[0] == 0 && mask.at<Vec3b>(i, j)[1] == 0 && mask.at<Vec3b>(i, j)[2] == 0) {
                img.at<Vec3b>(i, j)[0] = 0;
                img.at<Vec3b>(i, j)[1] = 255;
                img.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }

    imshow("Change background color", img);
    waitKey(0);

    //bitwise_and(mask, img, img);
    //imshow("After bitwise AND", img);
    //imshow("After changing background color", img);

    vector<vector<Point> > contours_poly( contours.size() );
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<float>radius( contours.size() );
    vector<float>area( contours.size() );
    vector<Point2f>center( contours.size() );

    int max_area;
    int max_x;
    int max_y;
    int max_height;
    int max_width;


    //Find rectangle with max area
    for( size_t i = 0; i < contours.size(); i++ ){
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );                //Poly approximation
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );                       //Squares for countures
        //minEnclosingCircle( contours_poly[i], center[i], radius[i] );               //Circles for countures
        area[i]= contourArea(Mat(contours_poly[i]));                                //Area ???

        int pom = boundRect[i].width * boundRect[i].height;

        if(i == 0){
            max_area = pom;
            max_x = boundRect[i].x;
            max_y = boundRect[i].y;
            max_height = boundRect[i].height;
            max_width = boundRect[i].width;
        }else if(pom > max_area){
            max_area = pom;
            max_x = boundRect[i].x;
            max_y = boundRect[i].y;
            max_height = boundRect[i].height;
            max_width = boundRect[i].width;
        }
    }


    //Drawing found countures on image

    //Black image
    Mat drawing = Mat::zeros( imgToProcess.size(), CV_8UC3 );

    //Contoures
    for( int i = 0; i< contours.size(); i++ ){
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );

        double AreaContour = contourArea(contours[i]);
        if(AreaContour < MaxAreaContour && AreaContour > MinAreaContour){
            //drawContours(mask,contours, i , Scalar(255,255,255),cv::FILLED);
            drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        }

        //ellipse( drawing, minEllipse[i], color, 2, 8 );// ellipse
        //TODO dodaj za pravougaonike ogranicenje
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
    }

    imshow("Contour", drawing);
    waitKey(0);

    //Crop image

    //first rows then columns
    Mat cropped_image = img(Range(max_y, (max_y + max_height)), Range(max_x, (max_x + max_width)));
    imshow("Cropped image", cropped_image);
    waitKey(0);

    //Segmentation
    int Clusters = 4;
    Mat clustered_Image = K_Means(cropped_image, Clusters);

    imshow("Segmented image", clustered_Image);

//    Save output image
//    imwrite("Clustered_Image.png", Clustered_Image);
//    system("pause");

    waitKey();
    return 0;
}

Mat K_Means(Mat Input, int K) {
    Mat samples(Input.rows * Input.cols, Input.channels(), CV_32F);
    for (int y = 0; y < Input.rows; y++)
        for (int x = 0; x < Input.cols; x++)
            for (int z = 0; z < Input.channels(); z++)
                if (Input.channels() == 3) {
                    samples.at<float>(y + x * Input.rows, z) = Input.at<Vec3b>(y, x)[z];
                }
                else {
                    samples.at<float>(y + x * Input.rows, z) = Input.at<uchar>(y, x);
                }

    Mat labels;
    int attempts = 5;
    Mat centers;
    kmeans(samples, K, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


    Mat new_image(Input.size(), Input.type());
    for (int y = 0; y < Input.rows; y++)
        for (int x = 0; x < Input.cols; x++)
        {
            int cluster_idx = labels.at<int>(y + x * Input.rows, 0);
            if (Input.channels()==3) {
                for (int i = 0; i < Input.channels(); i++) {
                    new_image.at<Vec3b>(y, x)[i] = centers.at<float>(cluster_idx, i);
                }
            }
            else {
                new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
            }
        }
    return new_image;
}