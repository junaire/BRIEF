#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "timer.h"

enum { PATCH_SIZE = 48, KERNEL_SIZE = 9 };

constexpr int bytes = 32;
constexpr bool use_orientation = false;

inline int smoothedSum(const cv::Mat &sum, const cv::KeyPoint &pt, int y, int x,
                       bool use_orientation, cv::Matx21f R) {
  static const int HALF_KERNEL = KERNEL_SIZE / 2;

  if (use_orientation) {
    int rx = (int)(((float)x) * R(1, 0) - ((float)y) * R(0, 0));
    int ry = (int)(((float)x) * R(0, 0) + ((float)y) * R(1, 0));
    if (rx > 24) rx = 24;
    if (rx < -24) rx = -24;
    if (ry > 24) ry = 24;
    if (ry < -24) ry = -24;
    x = rx;
    y = ry;
  }
  const int img_y = (int)(pt.pt.y + 0.5) + y;
  const int img_x = (int)(pt.pt.x + 0.5) + x;
  int r = sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1) -
          sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL) -
          sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1) +
          sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
  // printf("(%d, %d) = %d\n", y, x, r);
  return r;
}

struct RoiPredicate {
  RoiPredicate(const cv::Rect &_r) : r(_r) {}

  bool operator()(const cv::KeyPoint &keyPt) const {
    return !r.contains(keyPt.pt);
  }

  cv::Rect r;
};

inline void runByImageBorder(std::vector<cv::KeyPoint> &keypoints,
                             cv::Size imageSize, int borderSize) {
  if (borderSize > 0) {
    if (imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2)
      keypoints.clear();
    else
      keypoints.erase(
          std::remove_if(
              keypoints.begin(), keypoints.end(),
              RoiPredicate(cv::Rect(cv::Point(borderSize, borderSize),
                                    cv::Point(imageSize.width - borderSize,
                                              imageSize.height - borderSize)))),
          keypoints.end());
  }
}

template <typename F>
void ComputeBRIEFCommon(cv::InputArray image,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors, F &&test_fn) {
  cv::Mat sum;

  cv::Mat grayImage = image.getMat();
  if (image.type() != CV_8U) {
    cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
  }

  // {
  //   Timer _("cv::integral");
  cv::integral(grayImage, sum, CV_32S);
  // }

  // {
  //   Timer _("runByImageBorder");
  runByImageBorder(keypoints, image.size(), PATCH_SIZE / 2 + KERNEL_SIZE / 2);
  // }

  // Timer _("test fn");
  descriptors.create((int)keypoints.size(), bytes, CV_8U);
  descriptors.setTo(cv::Scalar::all(0));
  test_fn(sum, keypoints, descriptors, use_orientation);
}
