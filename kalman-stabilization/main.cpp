#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    KalmanFilter kalman(4, 2, 0);

    kalman.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    kalman.measurementMatrix = (Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    kalman.processNoiseCov = (Mat_<float>(4, 4) <<
        0.2, 0,    0,    0,
        0,    0.2, 0,    0,
        0,    0,    0.2, 0,
        0,    0,    0,    0.3);

    kalman.measurementNoiseCov = (Mat_<float>(2, 2) <<
        1e-1, 0,
        0,    1e-1);

    kalman.errorCovPost = Mat::eye(4, 4, CV_32F);

    Mat state(4, 1, CV_32F);
    state.at<float>(0) = 0; // x
    state.at<float>(1) = 0; // y
    state.at<float>(2) = 0; // vx
    state.at<float>(3) = 0; // vy
    kalman.statePost = state;

    VideoCapture cap("/Users/emre/Desktop/kalman/kalman-stabilization/1.avi");
    if (!cap.isOpened()) {
        cerr << "cant open" << endl;
        return -1;
    }

    Mat frame, prev_gray;
    cap >> frame;
    if (frame.empty()) {
        cerr << "cant read" << endl;
        return -1;
    }
    cvtColor(frame, prev_gray, COLOR_BGR2GRAY);

    vector<Point2f> prev_points;
    goodFeaturesToTrack(prev_gray, prev_points, 50, 0.01, 10);

    int frame_count = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (frame_count % 2 == 0) {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            vector<Point2f> curr_points;
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(prev_gray, gray, prev_points, curr_points, status, err);

            Mat measurement(2, 1, CV_32F, Scalar(0));
            int valid_points = 0;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i]) {
                    float dx = curr_points[i].x - prev_points[i].x;
                    float dy = curr_points[i].y - prev_points[i].y;

                    float max_move = 500.0; //
                    if (fabs(dx) > max_move || fabs(dy) > max_move) {
                        continue;
                    }

                    measurement.at<float>(0) += dx;
                    measurement.at<float>(1) += dy;
                    valid_points++;
                }
            }

            if (valid_points > 0) {
                measurement.at<float>(0) /= valid_points;
                measurement.at<float>(1) /= valid_points;

                kalman.correct(measurement);
                Mat prediction = kalman.predict();

                int dx = static_cast<int>(prediction.at<float>(0));
                int dy = static_cast<int>(prediction.at<float>(1));

                Mat stabilized_frame;
                Mat translation = (Mat_<double>(2, 3) << 1, 0, -dx, 0, 1, -dy);
                warpAffine(frame, stabilized_frame, translation, frame.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

                imshow("original video", frame);
                imshow("stabilized video", stabilized_frame);
            }

            gray.copyTo(prev_gray);
            goodFeaturesToTrack(prev_gray, prev_points, 50, 0.01, 10);
        }

        frame_count++;

        if (waitKey(30) == 27) break;
    }

    return 0;
}
