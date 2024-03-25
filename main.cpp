#include <cassert>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "timer.h"

void ComputeBRIEFCPU(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints,
                     cv::OutputArray descriptors);
// BRIEF descriptor implemented on GPU with CUDA.
void ComputeBRIEFGPU(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints,
                     cv::OutputArray descriptors);

void ShowDescriptorMatches(const cv::Mat &img1, const cv::Mat &img2,
                           const std::vector<cv::KeyPoint> &key_pts1,
                           const std::vector<cv::KeyPoint> &key_pts2,
                           const cv::Mat &descriptor1,
                           const cv::Mat &descriptor2,
                           const std::string &title) {
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
  std::vector<cv::DMatch> matchers;
  matcher->match(descriptor1, descriptor2, matchers);

  cv::Mat matches;
  cv::drawMatches(img1, key_pts1, img2, key_pts2, matchers, matches);

  cv::namedWindow(title, cv::WINDOW_NORMAL);
  cv::resizeWindow(title, 800, 600);
  cv::imshow(title, matches);
}

std::vector<cv::KeyPoint> DetectBRIEF(const cv::Mat &img) {
  auto detector = cv::xfeatures2d::StarDetector::create();
  std::vector<cv::KeyPoint> key_pts;
  detector->detect(img, key_pts);
  return key_pts;
}

template <typename ComputeFunc>
cv::Mat DetectAndComputeBRIEF(const cv::Mat &img, ComputeFunc func) {
  std::vector<cv::KeyPoint> key_pts = DetectBRIEF(img);
  cv::Mat descriptor;

  func(img, key_pts, descriptor);

  return descriptor;
}

void TestResults(const cv::Mat &descriptor1, const cv::Mat &descriptor2) {
  assert(std::equal(descriptor1.begin<uchar>(), descriptor1.end<uchar>(),
                    descriptor2.begin<uchar>()) &&
         "Descriptors differs!");
  printf("OK!\n");
}

void ComputeBRIEFOpenCV(const cv::Mat &img, std::vector<cv::KeyPoint> &key_pts,
                        cv::OutputArray descriptor) {
  auto d = cv::xfeatures2d::BriefDescriptorExtractor::create();
  d->compute(img, key_pts, descriptor);
}

int main(int argv, char **argc) {
  cv::Mat img = cv::imread("assets/1.jpg");

  cv::Mat descriptor1, descriptor2, descriptor3;
  {
    Timer _("OpenCV");
    descriptor1 = DetectAndComputeBRIEF(img, ComputeBRIEFOpenCV);
  }
  {
    Timer _("CPU from scratch");
    descriptor2 = DetectAndComputeBRIEF(img, ComputeBRIEFCPU);
  }
  {
    Timer _("GPU with Cuda");
    descriptor3 = DetectAndComputeBRIEF(img, ComputeBRIEFGPU);
  }
  TestResults(descriptor1, descriptor2);
  TestResults(descriptor1, descriptor3);
}
