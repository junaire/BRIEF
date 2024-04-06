import numpy as np
import cv2
import re
from pygame import Rect

PATCH_SIZE = 48  # 对每一个关键点，在48*48矩形区域内，形成比较点对
KERNEL_SIZE = 9  # 对每一个比较点，求该点9*9区域内的像素和


def get_pt_lines(rawstr):  # 从原始txt文档中，获取比较点对
    lines = rawstr.split(";")
    pt_bytes = []
    for l in lines:  # 对于所有行中的每一行
        pt_byte = get_pt_line(l)  # 提取每一行的点对
        if len(pt_byte) != 0:
            pt_bytes.append(pt_byte)  # 保存点对
    return pt_bytes


def get_pt_line(
    line,
):  # 从原始文件的每一行中，提取出比较点对，每一行8个点对，对应一个字节
    line = line.replace(" ", "")  # 删除空格
    blocks = line.split("+")  # 按‘+’进行分割
    pts = []
    for b in blocks:
        str1 = re.findall("\(SMOOTHED\((.+?)\)<SMOOTHED", b)  # 匹配关键字副
        str2 = re.findall("<SMOOTHED\((.+?)\)\)", b)  # 匹配关键字副
        if str1 != [] and str2 != []:
            str1 = str1[0].split(",")
            str2 = str2[0].split(",")
            num1 = np.array(str1).astype(int)  # 得到左边数字
            num2 = np.array(str2).astype(int)  # 得到右边数字
            pts.append((num1, num2))  # 保存点对
    return pts  # 返回点对数组，8个点对


def runByImageBorder(kps, img_src, border_size):  # 剔除越界的关键点
    r = Rect(
        border_size,
        border_size,
        img_src.shape[1] - border_size,
        img_src.shape[0] - border_size,
    )
    newkp = []
    for kp in kps:
        x, y = kp.pt
        if x >= r[0] and y >= r[1] and x < r[2] and y < r[3]:
            newkp.append(kp)
    return newkp


def get_sum_from_integral(x, y, img_sum):
    # 以x，y为中心，选择矩形框，求该矩形区域内所有像素点之和
    # 矩形框左上和右下的坐标为([-HALF_KERNEL,-HALF_KERNEL],[HALF_KERNEL+1,HALF_KERNEL+])(相对于中心点的偏移坐标)
    # 因输入图像已经是积分图像，所以原图该区域的像素点之和，等同于，在积分图像上对该矩形四个顶点处像素的加减运算
    x, y = int(x), int(y)
    HALF_KERNEL = (int)(KERNEL_SIZE / 2)
    return (
        img_sum[y - HALF_KERNEL, x - HALF_KERNEL]
        + img_sum[y + HALF_KERNEL + 1, x + HALF_KERNEL + 1]
        - img_sum[y - HALF_KERNEL, x + HALF_KERNEL + 1]
        - img_sum[y + HALF_KERNEL + 1, x - HALF_KERNEL]
    )


def generated_des(kpx, kpy, sum_img, pt_arr):  # 为每一个关键点生成描述符
    byte_arr = []
    for (
        pt_byte
    ) in pt_arr:  # 对于所有点对（例如，32×8个点对）中的每一字节对点（1×8个点对）
        #print(len(pt_byte))
        byte_val = 0
        for bit in pt_byte:  # 对于每个字节点对（8个点对）中的每一点对
            #print(len(bit))
            byte_val = byte_val << 1
            left_offset, right_offset = bit  # 得到待比较的左、右点
            left_cx, left_cy = (
                kpx + 0.5 + left_offset[1],
                kpy + 0.5 + left_offset[0],
            )  # 得到左边中心点
            right_cx, right_cy = (
                kpx + 0.5 + right_offset[1],
                kpy + 0.5 + right_offset[0],
            )  # 得到右边中心点
            # 根据左、右中心点，获取相应区域的像素和 ，并进行大小比较
            bit = get_sum_from_integral(
                left_cx, left_cy, sum_img
            ) < get_sum_from_integral(right_cx, right_cy, sum_img)
            bit = bit.astype(int)
            byte_val = byte_val + bit  # 将bit保存至byte中
            print(f"byte_val = {byte_val}")
        byte_arr.append(byte_val)  # 保存该byte至byte数组中
    print(f"================ byte_arr = {byte_arr} =============")
    return np.array(byte_arr)  # 返回byte数组，例如32个bytes

def generated_des2(kpx, kpy, sum_img, pt_arr):  # 为每一个关键点生成描述符
    byte_arr = []
    for (
        pt_byte
    ) in pt_arr:  # 对于所有点对（例如，32×8个点对）中的每一字节对点（1×8个点对）
        byte_val = 0
        for bit in pt_byte:  # 对于每个字节点对（8个点对）中的每一点对
            byte_val = byte_val << 1
            bit = compare()
            byte_val = byte_val + bit  # 将bit保存至byte中
        byte_arr.append(byte_val)  # 保存该byte至byte数组中
    return np.array(byte_arr)  # 返回byte数组，例如32个bytes


def my_brief_des(img_src, kps_src, bytes=32):
    # 先读取文件，获得比较点对
    if bytes == 16:
        filename = "generated_16.i"
    else:
        filename = "generated_32.i"
    with open(filename) as f:
        rawtxt = f.read()
    pt_arr = get_pt_lines(rawtxt)
    # 生成积分图像，便于后续的加速运算
    sum_img = cv2.integral(img_src, sdepth=cv2.CV_32S)
    # 对关键点进行筛选，剔除边界之外的点
    kps = runByImageBorder(kps_src, img_src, PATCH_SIZE / 2 + KERNEL_SIZE / 2)
    # 为所有关键点生成描述符
    des = []
    for kp in kps:
        byte_arr = generated_des(
            kp.pt[0], kp.pt[1], sum_img, pt_arr
        )  # 计算每一个关键点的描述符
        des.append(byte_arr)
    return kps, np.array(des)


def main():
    img_src = cv2.imread("assets/1.jpg", cv2.IMREAD_GRAYSCALE)
    # 使用fast算子，寻找出图像的关键点
    fast = cv2.FastFeatureDetector_create(
        threshold=10, type=cv2.FastFeatureDetector_TYPE_9_16
    )
    fast.setNonmaxSuppression(True)
    fast_kps = fast.detect(img_src)
    # 使用自行编写的brief函数生成描述符
    my_kps, my_des = my_brief_des(img_src, fast_kps, bytes=32)
    print(my_des)


main()
