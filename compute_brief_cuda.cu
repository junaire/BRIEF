#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "compute_brief_common.h"

#define CHECK(call)                                                    \
  do {                                                                 \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess) {                                        \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
      printf("Code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                         \
    }                                                                  \
  } while (false)

__device__ int smoothedSumCuda(cv::cuda::PtrStepSz<int> sum,
                               const cv::KeyPoint &pt, int y, int x) {
  static const int HALF_KERNEL = KERNEL_SIZE / 2;
  const int img_y = (int)(pt.pt.y + 0.5) + y;
  const int img_x = (int)(pt.pt.x + 0.5) + x;
#define A(y, x) sum.ptr(y)[x]
  int r = A(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1) -
          A(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL) -
          A(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1) +
          A(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
  return r;
#undef A
}

__global__ void PixelTest32Kernel(
    cv::cuda::PtrStepSz<int> sum,
    cv::cuda::PtrStepSz<unsigned char> descriptors,
    cv::KeyPoint *__restrict__ keypoints, int num_keypoints) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_keypoints) {
    unsigned char *desc = descriptors.ptr(idx);
    const cv::KeyPoint &pt = keypoints[idx];
#define SMOOTHED(y, x) smoothedSumCuda(sum, pt, y, x)
    desc[0] = (uchar)(((SMOOTHED(-2, -1) < SMOOTHED(7, -1)) << 7) +
                      ((SMOOTHED(-14, -1) < SMOOTHED(-3, 3)) << 6) +
                      ((SMOOTHED(1, -2) < SMOOTHED(11, 2)) << 5) +
                      ((SMOOTHED(1, 6) < SMOOTHED(-10, -7)) << 4) +
                      ((SMOOTHED(13, 2) < SMOOTHED(-1, 0)) << 3) +
                      ((SMOOTHED(-14, 5) < SMOOTHED(5, -3)) << 2) +
                      ((SMOOTHED(-2, 8) < SMOOTHED(2, 4)) << 1) +
                      ((SMOOTHED(-11, 8) < SMOOTHED(-15, 5)) << 0));
    desc[1] = (uchar)(((SMOOTHED(-6, -23) < SMOOTHED(8, -9)) << 7) +
                      ((SMOOTHED(-12, 6) < SMOOTHED(-10, 8)) << 6) +
                      ((SMOOTHED(-3, -1) < SMOOTHED(8, 1)) << 5) +
                      ((SMOOTHED(3, 6) < SMOOTHED(5, 6)) << 4) +
                      ((SMOOTHED(-7, -6) < SMOOTHED(5, -5)) << 3) +
                      ((SMOOTHED(22, -2) < SMOOTHED(-11, -8)) << 2) +
                      ((SMOOTHED(14, 7) < SMOOTHED(8, 5)) << 1) +
                      ((SMOOTHED(-1, 14) < SMOOTHED(-5, -14)) << 0));
    desc[2] = (uchar)(((SMOOTHED(-14, 9) < SMOOTHED(2, 0)) << 7) +
                      ((SMOOTHED(7, -3) < SMOOTHED(22, 6)) << 6) +
                      ((SMOOTHED(-6, 6) < SMOOTHED(-8, -5)) << 5) +
                      ((SMOOTHED(-5, 9) < SMOOTHED(7, -1)) << 4) +
                      ((SMOOTHED(-3, -7) < SMOOTHED(-10, -18)) << 3) +
                      ((SMOOTHED(4, -5) < SMOOTHED(0, 11)) << 2) +
                      ((SMOOTHED(2, 3) < SMOOTHED(9, 10)) << 1) +
                      ((SMOOTHED(-10, 3) < SMOOTHED(4, 9)) << 0));
    desc[3] = (uchar)(((SMOOTHED(0, 12) < SMOOTHED(-3, 19)) << 7) +
                      ((SMOOTHED(1, 15) < SMOOTHED(-11, -5)) << 6) +
                      ((SMOOTHED(14, -1) < SMOOTHED(7, 8)) << 5) +
                      ((SMOOTHED(7, -23) < SMOOTHED(-5, 5)) << 4) +
                      ((SMOOTHED(0, -6) < SMOOTHED(-10, 17)) << 3) +
                      ((SMOOTHED(13, -4) < SMOOTHED(-3, -4)) << 2) +
                      ((SMOOTHED(-12, 1) < SMOOTHED(-12, 2)) << 1) +
                      ((SMOOTHED(0, 8) < SMOOTHED(3, 22)) << 0));
    desc[4] = (uchar)(((SMOOTHED(-13, 13) < SMOOTHED(3, -1)) << 7) +
                      ((SMOOTHED(-16, 17) < SMOOTHED(6, 10)) << 6) +
                      ((SMOOTHED(7, 15) < SMOOTHED(-5, 0)) << 5) +
                      ((SMOOTHED(2, -12) < SMOOTHED(19, -2)) << 4) +
                      ((SMOOTHED(3, -6) < SMOOTHED(-4, -15)) << 3) +
                      ((SMOOTHED(8, 3) < SMOOTHED(0, 14)) << 2) +
                      ((SMOOTHED(4, -11) < SMOOTHED(5, 5)) << 1) +
                      ((SMOOTHED(11, -7) < SMOOTHED(7, 1)) << 0));
    desc[5] = (uchar)(((SMOOTHED(6, 12) < SMOOTHED(21, 3)) << 7) +
                      ((SMOOTHED(-3, 2) < SMOOTHED(14, 1)) << 6) +
                      ((SMOOTHED(5, 1) < SMOOTHED(-5, 11)) << 5) +
                      ((SMOOTHED(3, -17) < SMOOTHED(-6, 2)) << 4) +
                      ((SMOOTHED(6, 8) < SMOOTHED(5, -10)) << 3) +
                      ((SMOOTHED(-14, -2) < SMOOTHED(0, 4)) << 2) +
                      ((SMOOTHED(5, -7) < SMOOTHED(-6, 5)) << 1) +
                      ((SMOOTHED(10, 4) < SMOOTHED(4, -7)) << 0));
    desc[6] = (uchar)(((SMOOTHED(22, 0) < SMOOTHED(7, -18)) << 7) +
                      ((SMOOTHED(-1, -3) < SMOOTHED(0, 18)) << 6) +
                      ((SMOOTHED(-4, 22) < SMOOTHED(-5, 3)) << 5) +
                      ((SMOOTHED(1, -7) < SMOOTHED(2, -3)) << 4) +
                      ((SMOOTHED(19, -20) < SMOOTHED(17, -2)) << 3) +
                      ((SMOOTHED(3, -10) < SMOOTHED(-8, 24)) << 2) +
                      ((SMOOTHED(-5, -14) < SMOOTHED(7, 5)) << 1) +
                      ((SMOOTHED(-2, 12) < SMOOTHED(-4, -15)) << 0));
    desc[7] = (uchar)(((SMOOTHED(4, 12) < SMOOTHED(0, -19)) << 7) +
                      ((SMOOTHED(20, 13) < SMOOTHED(3, 5)) << 6) +
                      ((SMOOTHED(-8, -12) < SMOOTHED(5, 0)) << 5) +
                      ((SMOOTHED(-5, 6) < SMOOTHED(-7, -11)) << 4) +
                      ((SMOOTHED(6, -11) < SMOOTHED(-3, -22)) << 3) +
                      ((SMOOTHED(15, 4) < SMOOTHED(10, 1)) << 2) +
                      ((SMOOTHED(-7, -4) < SMOOTHED(15, -6)) << 1) +
                      ((SMOOTHED(5, 10) < SMOOTHED(0, 24)) << 0));
    desc[8] = (uchar)(((SMOOTHED(3, 6) < SMOOTHED(22, -2)) << 7) +
                      ((SMOOTHED(-13, 14) < SMOOTHED(4, -4)) << 6) +
                      ((SMOOTHED(-13, 8) < SMOOTHED(-18, -22)) << 5) +
                      ((SMOOTHED(-1, -1) < SMOOTHED(-7, 3)) << 4) +
                      ((SMOOTHED(-19, -12) < SMOOTHED(4, 3)) << 3) +
                      ((SMOOTHED(8, 10) < SMOOTHED(13, -2)) << 2) +
                      ((SMOOTHED(-6, -1) < SMOOTHED(-6, -5)) << 1) +
                      ((SMOOTHED(2, -21) < SMOOTHED(-3, 2)) << 0));
    desc[9] = (uchar)(((SMOOTHED(4, -7) < SMOOTHED(0, 16)) << 7) +
                      ((SMOOTHED(-6, -5) < SMOOTHED(-12, -1)) << 6) +
                      ((SMOOTHED(1, -1) < SMOOTHED(9, 18)) << 5) +
                      ((SMOOTHED(-7, 10) < SMOOTHED(-11, 6)) << 4) +
                      ((SMOOTHED(4, 3) < SMOOTHED(19, -7)) << 3) +
                      ((SMOOTHED(-18, 5) < SMOOTHED(-4, 5)) << 2) +
                      ((SMOOTHED(4, 0) < SMOOTHED(-20, 4)) << 1) +
                      ((SMOOTHED(7, -11) < SMOOTHED(18, 12)) << 0));
    desc[10] = (uchar)(((SMOOTHED(-20, 17) < SMOOTHED(-18, 7)) << 7) +
                       ((SMOOTHED(2, 15) < SMOOTHED(19, -11)) << 6) +
                       ((SMOOTHED(-18, 6) < SMOOTHED(-7, 3)) << 5) +
                       ((SMOOTHED(-4, 1) < SMOOTHED(-14, 13)) << 4) +
                       ((SMOOTHED(17, 3) < SMOOTHED(2, -8)) << 3) +
                       ((SMOOTHED(-7, 2) < SMOOTHED(1, 6)) << 2) +
                       ((SMOOTHED(17, -9) < SMOOTHED(-2, 8)) << 1) +
                       ((SMOOTHED(-8, -6) < SMOOTHED(-1, 12)) << 0));
    desc[11] = (uchar)(((SMOOTHED(-2, 4) < SMOOTHED(-1, 6)) << 7) +
                       ((SMOOTHED(-2, 7) < SMOOTHED(6, 8)) << 6) +
                       ((SMOOTHED(-8, -1) < SMOOTHED(-7, -9)) << 5) +
                       ((SMOOTHED(8, -9) < SMOOTHED(15, 0)) << 4) +
                       ((SMOOTHED(0, 22) < SMOOTHED(-4, -15)) << 3) +
                       ((SMOOTHED(-14, -1) < SMOOTHED(3, -2)) << 2) +
                       ((SMOOTHED(-7, -4) < SMOOTHED(17, -7)) << 1) +
                       ((SMOOTHED(-8, -2) < SMOOTHED(9, -4)) << 0));
    desc[12] = (uchar)(((SMOOTHED(5, -7) < SMOOTHED(7, 7)) << 7) +
                       ((SMOOTHED(-5, 13) < SMOOTHED(-8, 11)) << 6) +
                       ((SMOOTHED(11, -4) < SMOOTHED(0, 8)) << 5) +
                       ((SMOOTHED(5, -11) < SMOOTHED(-9, -6)) << 4) +
                       ((SMOOTHED(2, -6) < SMOOTHED(3, -20)) << 3) +
                       ((SMOOTHED(-6, 2) < SMOOTHED(6, 10)) << 2) +
                       ((SMOOTHED(-6, -6) < SMOOTHED(-15, 7)) << 1) +
                       ((SMOOTHED(-6, -3) < SMOOTHED(2, 1)) << 0));
    desc[13] = (uchar)(((SMOOTHED(11, 0) < SMOOTHED(-3, 2)) << 7) +
                       ((SMOOTHED(7, -12) < SMOOTHED(14, 5)) << 6) +
                       ((SMOOTHED(0, -7) < SMOOTHED(-1, -1)) << 5) +
                       ((SMOOTHED(-16, 0) < SMOOTHED(6, 8)) << 4) +
                       ((SMOOTHED(22, 11) < SMOOTHED(0, -3)) << 3) +
                       ((SMOOTHED(19, 0) < SMOOTHED(5, -17)) << 2) +
                       ((SMOOTHED(-23, -14) < SMOOTHED(-13, -19)) << 1) +
                       ((SMOOTHED(-8, 10) < SMOOTHED(-11, -2)) << 0));
    desc[14] = (uchar)(((SMOOTHED(-11, 6) < SMOOTHED(-10, 13)) << 7) +
                       ((SMOOTHED(1, -7) < SMOOTHED(14, 0)) << 6) +
                       ((SMOOTHED(-12, 1) < SMOOTHED(-5, -5)) << 5) +
                       ((SMOOTHED(4, 7) < SMOOTHED(8, -1)) << 4) +
                       ((SMOOTHED(-1, -5) < SMOOTHED(15, 2)) << 3) +
                       ((SMOOTHED(-3, -1) < SMOOTHED(7, -10)) << 2) +
                       ((SMOOTHED(3, -6) < SMOOTHED(10, -18)) << 1) +
                       ((SMOOTHED(-7, -13) < SMOOTHED(-13, 10)) << 0));
    desc[15] = (uchar)(((SMOOTHED(1, -1) < SMOOTHED(13, -10)) << 7) +
                       ((SMOOTHED(-19, 14) < SMOOTHED(8, -14)) << 6) +
                       ((SMOOTHED(-4, -13) < SMOOTHED(7, 1)) << 5) +
                       ((SMOOTHED(1, -2) < SMOOTHED(12, -7)) << 4) +
                       ((SMOOTHED(3, -5) < SMOOTHED(1, -5)) << 3) +
                       ((SMOOTHED(-2, -2) < SMOOTHED(8, -10)) << 2) +
                       ((SMOOTHED(2, 14) < SMOOTHED(8, 7)) << 1) +
                       ((SMOOTHED(3, 9) < SMOOTHED(8, 2)) << 0));
    desc[16] = (uchar)(((SMOOTHED(-9, 1) < SMOOTHED(-18, 0)) << 7) +
                       ((SMOOTHED(4, 0) < SMOOTHED(1, 12)) << 6) +
                       ((SMOOTHED(0, 9) < SMOOTHED(-14, -10)) << 5) +
                       ((SMOOTHED(-13, -9) < SMOOTHED(-2, 6)) << 4) +
                       ((SMOOTHED(1, 5) < SMOOTHED(10, 10)) << 3) +
                       ((SMOOTHED(-3, -6) < SMOOTHED(-16, -5)) << 2) +
                       ((SMOOTHED(11, 6) < SMOOTHED(-5, 0)) << 1) +
                       ((SMOOTHED(-23, 10) < SMOOTHED(1, 2)) << 0));
    desc[17] = (uchar)(((SMOOTHED(13, -5) < SMOOTHED(-3, 9)) << 7) +
                       ((SMOOTHED(-4, -1) < SMOOTHED(-13, -5)) << 6) +
                       ((SMOOTHED(10, 13) < SMOOTHED(-11, 8)) << 5) +
                       ((SMOOTHED(19, 20) < SMOOTHED(-9, 2)) << 4) +
                       ((SMOOTHED(4, -8) < SMOOTHED(0, -9)) << 3) +
                       ((SMOOTHED(-14, 10) < SMOOTHED(15, 19)) << 2) +
                       ((SMOOTHED(-14, -12) < SMOOTHED(-10, -3)) << 1) +
                       ((SMOOTHED(-23, -3) < SMOOTHED(17, -2)) << 0));
    desc[18] = (uchar)(((SMOOTHED(-3, -11) < SMOOTHED(6, -14)) << 7) +
                       ((SMOOTHED(19, -2) < SMOOTHED(-4, 2)) << 6) +
                       ((SMOOTHED(-5, 5) < SMOOTHED(3, -13)) << 5) +
                       ((SMOOTHED(2, -2) < SMOOTHED(-5, 4)) << 4) +
                       ((SMOOTHED(17, 4) < SMOOTHED(17, -11)) << 3) +
                       ((SMOOTHED(-7, -2) < SMOOTHED(1, 23)) << 2) +
                       ((SMOOTHED(8, 13) < SMOOTHED(1, -16)) << 1) +
                       ((SMOOTHED(-13, -5) < SMOOTHED(1, -17)) << 0));
    desc[19] = (uchar)(((SMOOTHED(4, 6) < SMOOTHED(-8, -3)) << 7) +
                       ((SMOOTHED(-5, -9) < SMOOTHED(-2, -10)) << 6) +
                       ((SMOOTHED(-9, 0) < SMOOTHED(-7, -2)) << 5) +
                       ((SMOOTHED(5, 0) < SMOOTHED(5, 2)) << 4) +
                       ((SMOOTHED(-4, -16) < SMOOTHED(6, 3)) << 3) +
                       ((SMOOTHED(2, -15) < SMOOTHED(-2, 12)) << 2) +
                       ((SMOOTHED(4, -1) < SMOOTHED(6, 2)) << 1) +
                       ((SMOOTHED(1, 1) < SMOOTHED(-2, -8)) << 0));
    desc[20] = (uchar)(((SMOOTHED(-2, 12) < SMOOTHED(-5, -2)) << 7) +
                       ((SMOOTHED(-8, 8) < SMOOTHED(-9, 9)) << 6) +
                       ((SMOOTHED(2, -10) < SMOOTHED(3, 1)) << 5) +
                       ((SMOOTHED(-4, 10) < SMOOTHED(-9, 4)) << 4) +
                       ((SMOOTHED(6, 12) < SMOOTHED(2, 5)) << 3) +
                       ((SMOOTHED(-3, -8) < SMOOTHED(0, 5)) << 2) +
                       ((SMOOTHED(-13, 1) < SMOOTHED(-7, 2)) << 1) +
                       ((SMOOTHED(-1, -10) < SMOOTHED(7, -18)) << 0));
    desc[21] = (uchar)(((SMOOTHED(-1, 8) < SMOOTHED(-9, -10)) << 7) +
                       ((SMOOTHED(-23, -1) < SMOOTHED(6, 2)) << 6) +
                       ((SMOOTHED(-5, -3) < SMOOTHED(3, 2)) << 5) +
                       ((SMOOTHED(0, 11) < SMOOTHED(-4, -7)) << 4) +
                       ((SMOOTHED(15, 2) < SMOOTHED(-10, -3)) << 3) +
                       ((SMOOTHED(-20, -8) < SMOOTHED(-13, 3)) << 2) +
                       ((SMOOTHED(-19, -12) < SMOOTHED(5, -11)) << 1) +
                       ((SMOOTHED(-17, -13) < SMOOTHED(-3, 2)) << 0));
    desc[22] = (uchar)(((SMOOTHED(7, 4) < SMOOTHED(-12, 0)) << 7) +
                       ((SMOOTHED(5, -1) < SMOOTHED(-14, -6)) << 6) +
                       ((SMOOTHED(-4, 11) < SMOOTHED(0, -4)) << 5) +
                       ((SMOOTHED(3, 10) < SMOOTHED(7, -3)) << 4) +
                       ((SMOOTHED(13, 21) < SMOOTHED(-11, 6)) << 3) +
                       ((SMOOTHED(-12, 24) < SMOOTHED(-7, -4)) << 2) +
                       ((SMOOTHED(4, 16) < SMOOTHED(3, -14)) << 1) +
                       ((SMOOTHED(-3, 5) < SMOOTHED(-7, -12)) << 0));
    desc[23] = (uchar)(((SMOOTHED(0, -4) < SMOOTHED(7, -5)) << 7) +
                       ((SMOOTHED(-17, -9) < SMOOTHED(13, -7)) << 6) +
                       ((SMOOTHED(22, -6) < SMOOTHED(-11, 5)) << 5) +
                       ((SMOOTHED(2, -8) < SMOOTHED(23, -11)) << 4) +
                       ((SMOOTHED(7, -10) < SMOOTHED(-1, 14)) << 3) +
                       ((SMOOTHED(-3, -10) < SMOOTHED(8, 3)) << 2) +
                       ((SMOOTHED(-13, 1) < SMOOTHED(-6, 0)) << 1) +
                       ((SMOOTHED(-7, -21) < SMOOTHED(6, -14)) << 0));
    desc[24] = (uchar)(((SMOOTHED(18, 19) < SMOOTHED(-4, -6)) << 7) +
                       ((SMOOTHED(10, 7) < SMOOTHED(-1, -4)) << 6) +
                       ((SMOOTHED(-1, 21) < SMOOTHED(1, -5)) << 5) +
                       ((SMOOTHED(-10, 6) < SMOOTHED(-11, -2)) << 4) +
                       ((SMOOTHED(18, -3) < SMOOTHED(-1, 7)) << 3) +
                       ((SMOOTHED(-3, -9) < SMOOTHED(-5, 10)) << 2) +
                       ((SMOOTHED(-13, 14) < SMOOTHED(17, -3)) << 1) +
                       ((SMOOTHED(11, -19) < SMOOTHED(-1, -18)) << 0));
    desc[25] = (uchar)(((SMOOTHED(8, -2) < SMOOTHED(-18, -23)) << 7) +
                       ((SMOOTHED(0, -5) < SMOOTHED(-2, -9)) << 6) +
                       ((SMOOTHED(-4, -11) < SMOOTHED(2, -8)) << 5) +
                       ((SMOOTHED(14, 6) < SMOOTHED(-3, -6)) << 4) +
                       ((SMOOTHED(-3, 0) < SMOOTHED(-15, 0)) << 3) +
                       ((SMOOTHED(-9, 4) < SMOOTHED(-15, -9)) << 2) +
                       ((SMOOTHED(-1, 11) < SMOOTHED(3, 11)) << 1) +
                       ((SMOOTHED(-10, -16) < SMOOTHED(-7, 7)) << 0));
    desc[26] = (uchar)(((SMOOTHED(-2, -10) < SMOOTHED(-10, -2)) << 7) +
                       ((SMOOTHED(-5, -3) < SMOOTHED(5, -23)) << 6) +
                       ((SMOOTHED(13, -8) < SMOOTHED(-15, -11)) << 5) +
                       ((SMOOTHED(-15, 11) < SMOOTHED(6, -6)) << 4) +
                       ((SMOOTHED(-16, -3) < SMOOTHED(-2, 2)) << 3) +
                       ((SMOOTHED(6, 12) < SMOOTHED(-16, 24)) << 2) +
                       ((SMOOTHED(-10, 0) < SMOOTHED(8, 11)) << 1) +
                       ((SMOOTHED(-7, 7) < SMOOTHED(-19, -7)) << 0));
    desc[27] = (uchar)(((SMOOTHED(5, 16) < SMOOTHED(9, -3)) << 7) +
                       ((SMOOTHED(9, 7) < SMOOTHED(-7, -16)) << 6) +
                       ((SMOOTHED(3, 2) < SMOOTHED(-10, 9)) << 5) +
                       ((SMOOTHED(21, 1) < SMOOTHED(8, 7)) << 4) +
                       ((SMOOTHED(7, 0) < SMOOTHED(1, 17)) << 3) +
                       ((SMOOTHED(-8, 12) < SMOOTHED(9, 6)) << 2) +
                       ((SMOOTHED(11, -7) < SMOOTHED(-8, -6)) << 1) +
                       ((SMOOTHED(19, 0) < SMOOTHED(9, 3)) << 0));
    desc[28] = (uchar)(((SMOOTHED(1, -7) < SMOOTHED(-5, -11)) << 7) +
                       ((SMOOTHED(0, 8) < SMOOTHED(-2, 14)) << 6) +
                       ((SMOOTHED(12, -2) < SMOOTHED(-15, -6)) << 5) +
                       ((SMOOTHED(4, 12) < SMOOTHED(0, -21)) << 4) +
                       ((SMOOTHED(17, -4) < SMOOTHED(-6, -7)) << 3) +
                       ((SMOOTHED(-10, -9) < SMOOTHED(-14, -7)) << 2) +
                       ((SMOOTHED(-15, -10) < SMOOTHED(-15, -14)) << 1) +
                       ((SMOOTHED(-7, -5) < SMOOTHED(5, -12)) << 0));
    desc[29] = (uchar)(((SMOOTHED(-4, 0) < SMOOTHED(15, -4)) << 7) +
                       ((SMOOTHED(5, 2) < SMOOTHED(-6, -23)) << 6) +
                       ((SMOOTHED(-4, -21) < SMOOTHED(-6, 4)) << 5) +
                       ((SMOOTHED(-10, 5) < SMOOTHED(-15, 6)) << 4) +
                       ((SMOOTHED(4, -3) < SMOOTHED(-1, 5)) << 3) +
                       ((SMOOTHED(-4, 19) < SMOOTHED(-23, -4)) << 2) +
                       ((SMOOTHED(-4, 17) < SMOOTHED(13, -11)) << 1) +
                       ((SMOOTHED(1, 12) < SMOOTHED(4, -14)) << 0));
    desc[30] = (uchar)(((SMOOTHED(-11, -6) < SMOOTHED(-20, 10)) << 7) +
                       ((SMOOTHED(4, 5) < SMOOTHED(3, 20)) << 6) +
                       ((SMOOTHED(-8, -20) < SMOOTHED(3, 1)) << 5) +
                       ((SMOOTHED(-19, 9) < SMOOTHED(9, -3)) << 4) +
                       ((SMOOTHED(18, 15) < SMOOTHED(11, -4)) << 3) +
                       ((SMOOTHED(12, 16) < SMOOTHED(8, 7)) << 2) +
                       ((SMOOTHED(-14, -8) < SMOOTHED(-3, 9)) << 1) +
                       ((SMOOTHED(-6, 0) < SMOOTHED(2, -4)) << 0));
    desc[31] = (uchar)(((SMOOTHED(1, -10) < SMOOTHED(-1, 2)) << 7) +
                       ((SMOOTHED(8, -7) < SMOOTHED(-6, 18)) << 6) +
                       ((SMOOTHED(9, 12) < SMOOTHED(-7, -23)) << 5) +
                       ((SMOOTHED(8, -6) < SMOOTHED(5, 2)) << 4) +
                       ((SMOOTHED(-9, 6) < SMOOTHED(-12, -7)) << 3) +
                       ((SMOOTHED(-1, -2) < SMOOTHED(-7, 2)) << 2) +
                       ((SMOOTHED(9, 9) < SMOOTHED(7, 15)) << 1) +
                       ((SMOOTHED(6, 2) < SMOOTHED(-6, 6)) << 0));
#undef SMOOTHED
  }
}

void pixelTests32Cuda(cv::InputArray _sum,
                      const std::vector<cv::KeyPoint> &keypoints,
                      cv::OutputArray _descriptors, bool use_orientation) {
  cudaEvent_t start, gpu_stop;

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&gpu_stop));
  CHECK(cudaEventRecord(start));
  cv::Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();

  cv::cuda::GpuMat gpu_sum(sum);
  cv::cuda::GpuMat gpu_descriptors(descriptors);

  cv::KeyPoint *gpu_keypoints;
  CHECK(cudaMalloc(&gpu_keypoints, keypoints.size() * sizeof(cv::KeyPoint)));
  CHECK(cudaMemcpy(gpu_keypoints, keypoints.data(),
                   keypoints.size() * sizeof(cv::KeyPoint),
                   cudaMemcpyHostToDevice));

  int block_size = 1024;
  int num_blocks = (keypoints.size() + block_size - 1) / block_size;
  PixelTest32Kernel<<<block_size, num_blocks>>>(
      gpu_sum, gpu_descriptors, gpu_keypoints, keypoints.size());

  gpu_descriptors.download(descriptors);
  gpu_sum.release();
  gpu_descriptors.release();
  CHECK(cudaFree(gpu_keypoints));
  CHECK(cudaEventRecord(gpu_stop));
  CHECK(cudaEventSynchronize(gpu_stop));
  float elapsed_gpu;
  CHECK(cudaEventElapsedTime(&elapsed_gpu, start, gpu_stop));
  printf("GPU time: %f ms\n", elapsed_gpu);
}
