## 算法步骤
1. 对图像进行灰度处理
2. 求图像的积分
3. 对关键点进行筛选，剔除边界之外的点
4. 测试像素点，得到描述子，伪代码如下：
```
def pixelTest:
    descriptor = []
    for pt in keypoints:
        arr = []
        for 8pp in pps: [[8] x 32]
            byte = 0
            for pp in 8pp:
                byte <<= 1
                left, right = pp
                img_y1 = pt.y + left.y + 0.5
                img_x1 = pt.x + left.x + 0.5

                img_y2 = pt.y + right.y + 0.5
                img_x2 = pt.x + right.x + 0.5

                bit = soothed_sum(img_y1, img_x1) < soothed_sum(img_y2, img_x2)
                byte += bit
            arr.append(byte)
        descriptor.append(arr)
    return descriptor
```

并行点：
第一层循环中，由于对每一个关键点的操作都是独立的，不存在数据依赖，所以可将其并行。每一个 CUDA 线程完成以下操作：
```
        idx = threadIdx.x + blockIdx.x * blockDim.x
        pt = keypoints[idx]
        arr = []
        for 8pp in pps: [[8] x 32]
            byte = 0
            for pp in 8pp:
                byte <<= 1
                left, right = pp
                img_y1 = pt.y + left.y + 0.5
                img_x1 = pt.x + left.x + 0.5

                img_y2 = pt.y + right.y + 0.5
                img_x2 = pt.x + right.x + 0.5

                bit = soothed_sum(img_y1, img_x1) < soothed_sum(img_y2, img_x2)
                byte += bit
            arr[idx] = byte
```