#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

int main() {

    string image_path = "/Users/emre/Desktop/kalman/kalman-tracking/soccerball";
    string annotation_path = "/Users/emre/Desktop/kalman/kalman-tracking/soccerball_mask";

    vector<string> images;
    vector<string> annotations;

    glob(image_path + "*jpg", images);
    glob(annotation_path + "*png", annotations);

    KalmanFilter kalman(4,2,0);
    Mat state(4,1, CV_32F);
    Mat measurement(2,1, CV_32F);

    if(!annotations.empty() && !images.empty()) {
        Mat first_mask = imread(annotations[0], IMREAD_GRAYSCALE);

        vector<vector<Point>> contours;
        findContours(first_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if(!contours.empty()) {
            Moments m = moments(contours[0]);
            state.at<float>(0) = m.m10/m.m00; // x loc
            state.at<float>(1) = m.m01/m.m00; // y loc
            state.at<float>(2) = 0; //velocity on x dir
            state.at<float>(3) = 0; //velocity on y dir

            kalman.transitionMatrix = (Mat_<float>(4,4)<<
                1,0,1,0,
                0,1,0,1,
                0,0,1,0,
                0,0,0,1);

            kalman.measurementMatrix = (Mat_<float>(2,4)<<
                1,0,0,0,
                0,1,0,0);

            kalman.processNoiseCov = (Mat_<float>(4,4)<<
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1);

            kalman.measurementNoiseCov = (Mat_<float>(2,2)<<
                1,0,
                0,1);

            kalman.errorCovPost = Mat::eye(4,4,CV_32F);
        }
    }

            for(size_t i=0; i<images.size(); ++i) {
                Mat image = imread(images[i]);
                Mat current_mask = imread(annotations[i], IMREAD_GRAYSCALE);

                vector<vector<Point>>current_contours;
                findContours(current_mask, current_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                if (!current_contours.empty()) {
                    Moments m = moments(current_contours[0]);
                    measurement.at<float>(0) = m.m10 / m.m00; // x loc
                    measurement.at<float>(1) = m.m01 / m.m00; // y loc
                } else {
                    //use preior one if theres no contour
                    measurement.at<float>(0) = state.at<float>(0);
                    measurement.at<float>(1) = state.at<float>(1);
                }

                kalman.correct(measurement);

                state = kalman.predict();

                cout << "predicted state: (" << state.at<float>(0) << ", " << state.at<float>(1) << ")" << endl;

                circle(image, Point(state.at<float>(0), state.at<float>(1)), 10, Scalar(0,0,255), -1);
                imshow("tracking", image);
                waitKey(300);
            }
            return 0;
        }