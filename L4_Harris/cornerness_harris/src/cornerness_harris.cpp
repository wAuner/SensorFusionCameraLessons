#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    

    // TODO: Your task is to locate local maxima in the Harris response matrix
    // and perform a non-maximum suppression (NMS) in a local neighborhood around
    // each maximum. The resulting coordinates shall be stored in a list of keypoints
    // of the type `vector<cv::KeyPoint>`.

    // loop over every pixel in the resulting harris response matrix
    // and check whether there are overlaps
    // keep only maximum if there are overlaps
    std::vector<cv::KeyPoint> keyPoints;
    float maxOverlap = 0.0;
    for (int row = 0; row < dst_norm.rows; row++)
    {
        for (int col = 0; col < dst_norm.cols; col++)
        {
            int pixelResponse = static_cast<int>(dst_norm.at<float>(row, col));
            // start checking if response exceeds minimum threshold
            if (pixelResponse > minResponse)
            {
                // create KeyPoint instance
                cv::KeyPoint newKeyPoint;
                // remember rows = y && cols = x, needs to be swapped for x,y coordinates
                newKeyPoint.pt = cv::Point2f(col, row);
                // area to consider for NMS
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = pixelResponse;

                bool overlap = false;

                // compare with other keyPoints to find maximum
                for (auto it = keyPoints.begin(); it != keyPoints.end(); ++it) {
                    float overlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    // if the points overlap, check which one has the higher respsonse
                    // choose the one with the high response
                    if (overlap > maxOverlap) {
                        overlap = true;
                        if (newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }

                // add if no overlap with other points
                if (!overlap) {
                    keyPoints.push_back(newKeyPoint);
                }
            }
        }
    }
    // show result
    cv::Mat nmsImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keyPoints, nmsImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("NMS", nmsImage);
    cv::waitKey(0);
}

int main()
{
    cornernessHarris();
}