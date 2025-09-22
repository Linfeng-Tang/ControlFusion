import os
from random import randint

import cv2
import numpy as np


def motion_blur_kernel(length, angle):
    """
    创建一个动态模糊核（运动模糊核）

    参数:
        length (int): 模糊核的长度，决定模糊的强度
        angle (float): 模糊的角度（0-180度）

    返回:
        kernel (numpy.ndarray): 运动模糊核
    """
    # 创建空的核
    kernel = np.zeros((length, length), dtype=np.float32)

    # 计算核的中心点
    center = length // 2
    kernel[center, :] = 1.0  # 水平线

    # 旋转核，形成指定角度的运动模糊
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))

    # 归一化核
    kernel /= kernel.sum()
    return kernel


def apply_motion_blur(image, kernel):
    """
    对输入图像应用运动模糊

    参数:
        image (numpy.ndarray): 输入图像
        kernel (numpy.ndarray): 模糊核

    返回:
        blurred (numpy.ndarray): 应用运动模糊后的图像
    """
    return cv2.filter2D(image, -1, kernel)


def process_images(input_dir, output_dir, blur_length=15, blur_angle=0):
    """
    读取指定目录下的所有图片，添加动态模糊，并保存到另一个目录

    参数:
        input_dir (str): 输入图像目录
        output_dir (str): 输出图像目录
        blur_length (int): 动态模糊核的长度
        blur_angle (float): 动态模糊的角度
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建模糊核
    kernel = motion_blur_kernel(blur_length, blur_angle)

    # 遍历输入目录下的所有图片
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 检查文件是否为图片
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # 读取图片
            image = cv2.imread(input_path)
            if image is None:
                print(f"无法读取文件: {input_path}")
                continue

            # 应用动态模糊
            blurred_image = apply_motion_blur(image, kernel)

            # 保存结果
            cv2.imwrite(output_path, blurred_image)
            print(f"已处理: {filename}")
        else:
            print(f"跳过非图片文件: {filename}")


if __name__ == "__main__":
    # 用户输入
    input_directory = ''
    output_directory = ''
    blur_length = 5  # 动态模糊核的长度（数值越大，模糊越明显）
    blur_angle = randint(20,20)  # 模糊的角度（0-180度，0度为水平模糊）

    process_images(input_directory, output_directory, blur_length, blur_angle)
