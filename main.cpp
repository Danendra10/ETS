#include <iostream>
#include <vector>
#include "ETS/main.h"
#include "png++/png.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include <opencv4/opencv2/highgui.hpp>

using namespace std;

int main(int argc, char **argv)
{
    try
    {
        // cv::namedWindow("Example 2-1", cv::WINDOW_NORMAL);
        cv::Mat image = cv::imread("/home/danendra/Kuliah/DSEC/ETS/data/2.jpeg");
        cv::Rect myROI(0, 0, 100, 100);
        cv::Mat croppedImage = image(myROI);

        cv::resize(image, image, cv::Size(30, 30));

        // thresh and get the contour of number
        // make a trackbar for hsv threshold
        // cv::createTrackbar("Hue Min", "Example 2-1", 0, 179);
        // cv::createTrackbar("Hue Max", "Example 2-1", 0, 179);
        // cv::createTrackbar("Sat Min", "Example 2-1", 0, 255);
        // cv::createTrackbar("Sat Max", "Example 2-1", 0, 255);
        // cv::createTrackbar("Val Min", "Example 2-1", 0, 255);
        // cv::createTrackbar("Val Max", "Example 2-1", 0, 255);

        cv::Mat imgHSV, imgThresholded;
        cv::cvtColor(image, imgHSV, cv::COLOR_BGR2HSV);
        // cv::imshow("HSV", imgHSV);

        while (true)
        {
            /*for thresholding purposes*/
            // int h_min = cv::getTrackbarPos("Hue Min", "Example 2-1");
            // int h_max = cv::getTrackbarPos("Hue Max", "Example 2-1");
            // int s_min = cv::getTrackbarPos("Sat Min", "Example 2-1");
            // int s_max = cv::getTrackbarPos("Sat Max", "Example 2-1");
            // int v_min = cv::getTrackbarPos("Val Min", "Example 2-1");
            // int v_max = cv::getTrackbarPos("Val Max", "Example 2-1");

            int h_min = 0;
            int h_max = 179;
            int s_min = 0;
            int s_max = 25;
            int v_min = 0;
            int v_max = 255;

            cv::Scalar lower(h_min, s_min, v_min);
            cv::Scalar upper(h_max, s_max, v_max);

            cv::inRange(imgHSV, lower, upper, imgThresholded);
            cv::bitwise_not(imgThresholded, imgThresholded);

            vector<vector<cv::Point>> contours;
            vector<cv::Vec4i> hierarchy;
            cv::findContours(imgThresholded, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
            cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2, 8, hierarchy, 0, cv::Point());
            cv::imshow("Thresholded Image", imgThresholded);
            cv::imshow("Original", image);
            // input = imgThresholded;
            // print img thresholded and insert to input
            // for (int i = 0; i < imgThresholded.rows; i++)
            // {
            //     for (int j = 0; j < imgThresholded.cols; j++)
            //     {
            //         cout << (int)imgThresholded.at<uchar>(i, j) << " ";
            //         // input[i].push_back((float)imgThresholded.at<uchar>(i, j));
            //     }
            //     cout << endl;
            // }
            // print the input vector
            normalized_input = NormalizationWithMat(imgThresholded, 0, 255);
            // printf("Size : %d %d\n", normalized_input.size(), normalized_input[0].size());
            weight1.resize(30);
            for (int i = 0; i < 30; i++)
                weight1[i].resize(30);
            bias1.resize(9);
            for (int i = 0; i < 9; i++)
                bias1[i].resize(1);

            RandomWeight(&weight1, normalized_input.size(), normalized_input[0].size());
            RandomBias(&bias1, 9, 1);

            output_sigmoid.resize(9);
            for (int i = 0; i < 9; i++)
                output_sigmoid[i].resize(1);
            output_sigmoid = NNLayer(normalized_input, weight1, bias1, 1);
            printf("Size : %d %d\n", output_sigmoid.size(), output_sigmoid[0].size());
            break;
            if (cv::waitKey(30) >= 0)
                break;
        }
    }
    catch (cv::Exception &e)
    {
        cout << e.what() << endl;
    }
}