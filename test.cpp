#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>

int main() {
    cv::Mat img = cv::imread("test.jpg");
    // BRIEF 本身并不提供任何查找特征的方法。这里我们使用 star 检测器。
    auto detector = cv::xfeatures2d::StarDetector::create();
    std::vector<cv::KeyPoint> key_pts;
    detector->detect(img, key_pts);
    auto d = cv::xfeatures2d::BriefDescriptorExtractor::create();
    cv::Mat descriptor;
    d->compute(img, key_pts, descriptor);
}