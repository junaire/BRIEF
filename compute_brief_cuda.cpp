#include "compute_brief_common.h"

void pixelTests32Cuda(cv::InputArray _sum,
                      const std::vector<cv::KeyPoint> &keypoints,
                      cv::OutputArray _descriptors, bool use_orientation);

void ComputeBRIEFGPU(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints,
                     cv::OutputArray descriptors) {
  ComputeBRIEFCommon(image, keypoints, descriptors, pixelTests32Cuda);
}