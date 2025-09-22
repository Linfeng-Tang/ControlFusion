import cv2
import os
import numpy as np


def adjust_contrast(image, alpha=0.5, beta=50):
    """
    使用线性变换调整图像的对比度。
    :param image: 输入的图像，灰度图像
    :param alpha: 对比度因子 (通常在 0.0 到 1.0 之间，0 会使图像完全暗淡)
    :param beta: 亮度偏移量 (通常在 0 到 100 之间，增大时图像会变亮)
    :return: 对比度调整后的图像
    """
    # 使用线性变换公式：new_image = alpha * image + beta
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def process_images_in_folder(folder_path, output_folder, alpha=0.5, beta=50):
    """
    读取文件夹中的所有红外图像，调整其对比度并保存到输出文件夹。
    :param folder_path: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param alpha: 对比度因子
    :param beta: 亮度偏移量
    """
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中所有图像文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # 遍历图像文件并处理每张图像
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像

        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 调整对比度
        adjusted_image = adjust_contrast(image, alpha, beta)

        # 保存处理后的图像
        output_image_path = os.path.join(output_folder, f"{image_file}")
        cv2.imwrite(output_image_path, adjusted_image)
        print(f"处理后的图像已保存: {output_image_path}")


if __name__ == "__main__":
    # 输入文件夹和输出文件夹路径
    input_folder = ''  # 请替换为你的输入文件夹路径
    output_folder = ''  # 请替换为你的输出文件夹路径

    # 调整对比度，alpha 和 beta 可以根据需求调整
    process_images_in_folder(input_folder, output_folder, alpha=0.5, beta=90)
    #只需要调整beta，越大，对比度越低
