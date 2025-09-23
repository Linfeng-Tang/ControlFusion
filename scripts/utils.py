import os
import sys
import random
import clip
import torch.nn.functional as F
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scripts.losses import fusion_prompt_loss


vi_noise_slight_prompt = "/media/lfw/Data/WYD_TextDataset/VI_Noise/VI_Noise_slight/train/text.txt"
assert os.path.exists(vi_noise_slight_prompt), "text prompt root: {} does not exist.".format(vi_noise_slight_prompt)
with open(vi_noise_slight_prompt, 'r', encoding='utf-8') as file:
        vi_noise_slight_lines = file.readlines()

vi_noise_moderate_prompt = "/media/lfw/Data/WYD_TextDataset/VI_Noise/VI_Noise_moderate/train/text.txt"
assert os.path.exists(vi_noise_moderate_prompt), "text prompt root: {} does not exist.".format(vi_noise_moderate_prompt)
with open(vi_noise_moderate_prompt, 'r', encoding='utf-8') as file:
        vi_noise_moderate_lines = file.readlines()

vi_noise_average_prompt = "/media/lfw/Data/WYD_TextDataset/VI_Noise/VI_Noise_average/train/text.txt"
assert os.path.exists(vi_noise_average_prompt), "text prompt root: {} does not exist.".format(vi_noise_average_prompt)
with open(vi_noise_average_prompt, 'r', encoding='utf-8') as file:
        vi_noise_average_lines = file.readlines()

vi_noise_extreme_prompt = "/media/lfw/Data/WYD_TextDataset/VI_Noise/VI_Noise_extreme/train/text.txt"
assert os.path.exists(vi_noise_extreme_prompt), "text prompt root: {} does not exist.".format(vi_noise_extreme_prompt)
with open(vi_noise_extreme_prompt, 'r', encoding='utf-8') as file:
        vi_noise_extreme_lines = file.readlines()


over_exposure_slight_prompt_path = "/media/lfw/Data/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_slight/train/text.txt"
assert os.path.exists(over_exposure_slight_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_slight_prompt_path)
with open(over_exposure_slight_prompt_path, 'r', encoding='utf-8') as file:
        over_exposure_slight_lines = file.readlines()

over_exposure_moderate_prompt_path = "/media/lfw/Data/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_moderate/train/text.txt"
assert os.path.exists(over_exposure_moderate_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_moderate_prompt_path)
with open(over_exposure_moderate_prompt_path, 'r', encoding='utf-8') as file:
        over_exposure_moderate_lines = file.readlines()

over_exposure_average_prompt_path = "/media/lfw/Data/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_average/train/text.txt"
assert os.path.exists(over_exposure_average_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_average_prompt_path)
with open(over_exposure_average_prompt_path, 'r', encoding='utf-8') as file:
        over_exposure_average_lines = file.readlines()

over_exposure_extreme_prompt_path = "/media/lfw/Data/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_extreme/train/text.txt"
assert os.path.exists(over_exposure_extreme_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_extreme_prompt_path)
with open(over_exposure_extreme_prompt_path, 'r', encoding='utf-8') as file:
        over_exposure_extreme_lines = file.readlines()


ir_low_contrast_slight_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_slight/train/text.txt"
assert os.path.exists(ir_low_contrast_slight_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_slight_prompt_path)
with open(ir_low_contrast_slight_prompt_path, 'r', encoding='utf-8') as file:
        ir_low_contrast_slight_lines = file.readlines()

ir_low_contrast_moderate_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_moderate/train/text.txt"
assert os.path.exists(ir_low_contrast_moderate_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_moderate_prompt_path)
with open(ir_low_contrast_moderate_prompt_path, 'r', encoding='utf-8') as file:
        ir_low_contrast_moderate_lines = file.readlines()

ir_low_contrast_average_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_average/train/text.txt"
assert os.path.exists(ir_low_contrast_average_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_average_prompt_path)
with open(ir_low_contrast_average_prompt_path, 'r', encoding='utf-8') as file:
        ir_low_contrast_average_lines = file.readlines()

ir_low_contrast_extreme_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_extreme/train/text.txt"
assert os.path.exists(ir_low_contrast_extreme_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_extreme_prompt_path)
with open(ir_low_contrast_extreme_prompt_path, 'r', encoding='utf-8') as file:
        ir_low_contrast_extreme_lines = file.readlines()


ir_noise_slight_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Noise/IR_Noise_slight/train/text.txt"
assert os.path.exists(ir_noise_slight_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_slight_prompt_path)
with open(ir_noise_slight_prompt_path, 'r', encoding='utf-8') as file:
        ir_noise_slight_lines = file.readlines()

ir_noise_moderate_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Noise/IR_Noise_moderate/train/text.txt"
assert os.path.exists(ir_noise_moderate_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_moderate_prompt_path)
with open(ir_noise_moderate_prompt_path, 'r', encoding='utf-8') as file:
        ir_noise_moderate_lines = file.readlines()

ir_noise_average_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Noise/IR_Noise_average/train/text.txt"
assert os.path.exists(ir_noise_average_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_average_prompt_path)
with open(ir_noise_average_prompt_path, 'r', encoding='utf-8') as file:
        ir_noise_average_lines = file.readlines()

ir_noise_extreme_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Noise/IR_Noise_extreme/train/text.txt"
assert os.path.exists(ir_noise_extreme_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_extreme_prompt_path)
with open(ir_noise_extreme_prompt_path, 'r', encoding='utf-8') as file:
        ir_noise_extreme_lines = file.readlines()


ir_stripe_noise_slight_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_slight/train/text.txt"
assert os.path.exists(ir_noise_slight_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_slight_prompt_path)
with open(ir_noise_slight_prompt_path, 'r', encoding='utf-8') as file:
        ir_stripe_noise_slight_lines = file.readlines()

ir_stripe_noise_moderate_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_moderate/train/text.txt"
assert os.path.exists(ir_noise_moderate_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_moderate_prompt_path)
with open(ir_noise_moderate_prompt_path, 'r', encoding='utf-8') as file:
        ir_stripe_noise_moderate_lines = file.readlines()

ir_stripe_noise_average_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_average/train/text.txt"
assert os.path.exists(ir_noise_average_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_average_prompt_path)
with open(ir_noise_average_prompt_path, 'r', encoding='utf-8') as file:
        ir_stripe_noise_average_lines = file.readlines()

ir_stripe_noise_extreme_prompt_path = "/media/lfw/Data/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_extreme/train/text.txt"
assert os.path.exists(ir_stripe_noise_extreme_prompt_path), "text prompt root: {} does not exist.".format(ir_stripe_noise_extreme_prompt_path)
with open(ir_stripe_noise_extreme_prompt_path, 'r', encoding='utf-8') as file:
        ir_stripe_noise_extreme_lines = file.readlines()



def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_visible_gt_path = []
    train_images_infrared_gt_path = []
    val_images_visible_path = []
    val_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型

    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root= os.path.join(train_root, "Infrared")

    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root= os.path.join(train_root, "Infrared_gt")

    val_visible_root = os.path.join(val_root, "Visible")
    val_infrared_root = os.path.join(val_root, "Infrared")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root)
                  if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible_root, i) for i in os.listdir(val_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_infrared_root, i) for i in os.listdir(val_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_visible_gt_path.sort()
    train_infrared_gt_path.sort()
    val_visible_path.sort()
    val_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))
    assert len(val_visible_path) == len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.\
                                          format(len(val_visible_path),len(val_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

        img_visible_gt_path=train_visible_gt_path[index]
        img_infrared_gt_path=train_infrared_gt_path[index]
        train_images_visible_gt_path.append(img_visible_gt_path)
        train_images_infrared_gt_path.append(img_infrared_gt_path)

    for index in range(len(val_visible_path)):
        img_visible_path=val_visible_path[index]
        img_infrared_path=val_infrared_path[index]
        val_images_visible_path.append(img_visible_path)
        val_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) + len(train_visible_gt_path) + len(train_infrared_gt_path) \
                         + len(val_visible_path) + len(val_infrared_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} visible gt images for training.".format(len(train_visible_gt_path)))
    print("{} infrared gt images for training.".format(len(train_infrared_gt_path)))
    print("{} visible images for validation.".format(len(val_visible_path)))
    print("{} infrared images for validation.\n".format(len(val_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path, train_visible_gt_path, train_infrared_gt_path]
    val_low_light_path_list = [val_visible_path, val_infrared_path]
    return train_low_light_path_list, val_low_light_path_list

def get_ir_low_contrast_slight_prompt():
    random_line = random.choice(ir_low_contrast_slight_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_low_contrast_moderate_prompt():
    random_line = random.choice(ir_low_contrast_moderate_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_low_contrast_average_prompt():
    random_line = random.choice(ir_low_contrast_average_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_low_contrast_extreme_prompt():
    random_line = random.choice(ir_low_contrast_extreme_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_noise_slight_prompt():
    random_line = random.choice(ir_noise_slight_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_noise_moderate_prompt():
    random_line = random.choice(ir_noise_moderate_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_noise_average_prompt():
    random_line = random.choice(ir_noise_average_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_noise_extreme_prompt():
    random_line = random.choice(ir_noise_extreme_lines)
    random_line = random_line.strip()
    return random_line

def get_over_exposure_slight_prompt():
    random_line = random.choice(over_exposure_slight_lines)
    random_line = random_line.strip()
    return random_line

def get_over_exposure_moderate_prompt():
    random_line = random.choice(over_exposure_moderate_lines)
    random_line = random_line.strip()
    return random_line

def get_over_exposure_average_prompt():
    random_line = random.choice(over_exposure_average_lines)
    random_line = random_line.strip()
    return random_line

def get_over_exposure_extreme_prompt():
    random_line = random.choice(over_exposure_extreme_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_stripe_noise_slight_prompt():
    random_line = random.choice(ir_stripe_noise_slight_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_stripe_noise_moderate_prompt():
    random_line = random.choice(ir_stripe_noise_moderate_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_stripe_noise_average_prompt():
    random_line = random.choice(ir_stripe_noise_average_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_stripe_noise_extreme_prompt():
    random_line = random.choice(ir_stripe_noise_extreme_lines)
    random_line = random_line.strip()
    return random_line

def get_vi_noise_slight_prompt():
    random_line = random.choice(vi_noise_slight_lines)
    random_line = random_line.strip()
    return random_line

def get_vi_noise_moderate_prompt():
    random_line = random.choice(vi_noise_moderate_lines)
    random_line = random_line.strip()
    return random_line

def get_vi_noise_average_prompt():
    random_line = random.choice(vi_noise_average_lines)
    random_line = random_line.strip()
    return random_line

def get_vi_noise_extreme_prompt():
    random_line = random.choice(vi_noise_extreme_lines)
    random_line = random_line.strip()
    return random_line

def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    model_clip.eval()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)


    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_line = []

        for index in range(len(task)):
        # default type degradation in vis image
            if task[index] == "ir_low_contrast_slight":
                text_line.append(get_ir_low_contrast_slight_prompt())
            elif task[index] == "ir_low_contrast_moderate":
                text_line.append(get_ir_low_contrast_moderate_prompt())
            elif task[index] == "ir_low_contrast_average":
                text_line.append(get_ir_low_contrast_average_prompt())
            elif task[index] == "ir_low_contrast_extreme":
                text_line.append(get_ir_low_contrast_extreme_prompt())
            elif task[index] == "over_exposure_slight":
                text_line.append(get_over_exposure_slight_prompt())
            elif task[index] == "over_exposure_moderate":
                text_line.append(get_over_exposure_moderate_prompt())
            elif task[index] == "over_exposure_average":
                text_line.append(get_over_exposure_average_prompt())
            elif task[index] == "over_exposure_extreme":
                text_line.append(get_over_exposure_extreme_prompt())
            elif task[index] == "ir_noise_slight":
                text_line.append(get_ir_noise_slight_prompt())
            elif task[index] == "ir_noise_moderate":
                text_line.append(get_ir_noise_moderate_prompt())
            elif task[index] == "ir_noise_average":
                text_line.append(get_ir_noise_average_prompt())
            elif task[index] == "ir_noise_extreme":
                text_line.append(get_ir_noise_extreme_prompt())
            elif task[index] == "ir_stripe_noise_slight":
                text_line.append(get_ir_stripe_noise_slight_prompt())
            elif task[index] == "ir_stripe_noise_moderate":
                text_line.append(get_ir_stripe_noise_moderate_prompt())
            elif task[index] == "ir_stripe_noise_average":
                text_line.append(get_ir_stripe_noise_average_prompt())
            elif task[index] == "ir_stripe_noise_extreme":
                text_line.append(get_ir_stripe_noise_extreme_prompt())
            elif task[index] == "vi_noise_slight":
                text_line.append(get_vi_noise_slight_prompt())
            elif task[index] == "vi_noise_moderate":
                text_line.append(get_vi_noise_moderate_prompt())
            elif task[index] == "vi_noise_average":
                text_line.append(get_vi_noise_average_prompt())
            elif task[index] == "vi_noise_extreme":
                text_line.append(get_vi_noise_extreme_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)
 

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        I_fused = model(I_A, I_B, text)

        ##########################################################################
        loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text


        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    loss_function_prompt = fusion_prompt_loss()

    model.eval()
    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)
    
    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "ir_low_contrast_slight":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the slight low contrast degradation.")
            elif task[index] == "ir_low_contrast_moderate":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the moderate low contrast degradation.")
            elif task[index] == "ir_low_contrast_average":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the average low contrast degradation.")
            elif task[index] == "ir_low_contrast_extreme":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the serious low contrast degradation.")
            elif task[index] == "over_exposure_slight":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the slight overexposure degradation.")
            elif task[index] == "over_exposure_moderate":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the moderate overexposure degradation.")
            elif task[index] == "over_exposure_average":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the average overexposure degradation.")
            elif task[index] == "over_exposure_extreme":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the serious overexposure degradation.")
            elif task[index] == "ir_noise_slight":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the slight noise degradation.")
            elif task[index] == "ir_noise_moderate":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the moderate noise degradation.")
            elif task[index] == "ir_noise_average":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the average noise degradation.")
            elif task[index] == "ir_noise_extreme":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the serious noise degradation.")
            elif task[index] == "ir_stripe_noise_slight":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the slight stripe noise degradation.")
            elif task[index] == "ir_stripe_noise_moderate":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the moderate stripe noise degradation.")
            elif task[index] == "ir_stripe_noise_average":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the average stripe noise degradation.")
            elif task[index] == "ir_stripe_noise_extreme":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the serious stripe noise degradation.")
            elif task[index] == "vi_noise_slight":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the slight noise degradation.")
            elif task[index] == "vi_noise_moderate":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the moderate noise degradation.")
            elif task[index] == "vi_noise_average":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the average noise degradation.")
            elif task[index] == "vi_noise_extreme":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the serious noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        I_fused = model(I_A, I_B, text)


        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text

        data_loader.desc = "[val epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1)

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)

def show_img(images,imagesl, B):
    for index in range(B):
        img = images[index, :]
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.imshow(img_np)
        img = imagesl[index, :]

        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(2)
        plt.imshow(img_np)
        plt.show(block=True)

def tensor2numpy(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_single(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3

def merge_and_print_paths(
    train_vi_noise_slight_path_list, val_vi_noise_slight_path_list,
    train_vi_noise_moderate_path_list, val_vi_noise_moderate_path_list,
    train_vi_noise_average_path_list, val_vi_noise_average_path_list,
    train_vi_noise_serious_path_list, val_vi_noise_serious_path_list,
    train_over_exposure_slight_path_list, val_over_exposure_slight_path_list,
    train_over_exposure_moderate_path_list, val_over_exposure_moderate_path_list,
    train_over_exposure_average_path_list, val_over_exposure_average_path_list,
    train_over_exposure_serious_path_list, val_over_exposure_serious_path_list,
    train_ir_low_contrast_slight_path_list, val_ir_low_contrast_slight_path_list,
    train_ir_low_contrast_moderate_path_list, val_ir_low_contrast_moderate_path_list,
    train_ir_low_contrast_average_path_list, val_ir_low_contrast_average_path_list,
    train_ir_low_contrast_serious_path_list, val_ir_low_contrast_serious_path_list,
    train_ir_noise_slight_path_list, val_ir_noise_slight_path_list,
    train_ir_noise_moderate_path_list, val_ir_noise_moderate_path_list,
    train_ir_noise_average_path_list, val_ir_noise_average_path_list,
    train_ir_noise_serious_path_list, val_ir_noise_serious_path_list,
    train_ir_stripe_noise_slight_path_list, val_ir_stripe_noise_slight_path_list,
    train_ir_stripe_noise_moderate_path_list, val_ir_stripe_noise_moderate_path_list,
    train_ir_stripe_noise_average_path_list, val_ir_stripe_noise_average_path_list,
    train_ir_stripe_noise_serious_path_list, val_ir_stripe_noise_serious_path_list
):
    # Combine paths for Visible and Infrared (VI) noise
    vi_noise_train_path_list = [
        train_vi_noise_slight_path_list,
        train_vi_noise_moderate_path_list,
        train_vi_noise_average_path_list,
        train_vi_noise_serious_path_list
    ]
    vi_noise_val_path_list = [
        val_vi_noise_slight_path_list,
        val_vi_noise_moderate_path_list,
        val_vi_noise_average_path_list,
        val_vi_noise_serious_path_list
    ]

    # Combine paths for Over-Exposure
    over_exposure_train_path_list = [
        train_over_exposure_slight_path_list,
        train_over_exposure_moderate_path_list,
        train_over_exposure_average_path_list,
        train_over_exposure_serious_path_list
    ]
    over_exposure_val_path_list = [
        val_over_exposure_slight_path_list,
        val_over_exposure_moderate_path_list,
        val_over_exposure_average_path_list,
        val_over_exposure_serious_path_list
    ]

    # Combine paths for IR Low Contrast
    ir_low_contrast_train_path_list = [
        train_ir_low_contrast_slight_path_list,
        train_ir_low_contrast_moderate_path_list,
        train_ir_low_contrast_average_path_list,
        train_ir_low_contrast_serious_path_list
    ]
    ir_low_contrast_val_path_list = [
        val_ir_low_contrast_slight_path_list,
        val_ir_low_contrast_moderate_path_list,
        val_ir_low_contrast_average_path_list,
        val_ir_low_contrast_serious_path_list
    ]

    # Combine paths for IR Noise
    ir_noise_train_path_list = [
        train_ir_noise_slight_path_list,
        train_ir_noise_moderate_path_list,
        train_ir_noise_average_path_list,
        train_ir_noise_serious_path_list
    ]
    ir_noise_val_path_list = [
        val_ir_noise_slight_path_list,
        val_ir_noise_moderate_path_list,
        val_ir_noise_average_path_list,
        val_ir_noise_serious_path_list
    ]

    # Combine paths for IR Stripe Noise
    ir_stripe_noise_train_path_list = [
        train_ir_stripe_noise_slight_path_list,
        train_ir_stripe_noise_moderate_path_list,
        train_ir_stripe_noise_average_path_list,
        train_ir_stripe_noise_serious_path_list
    ]
    ir_stripe_noise_val_path_list = [
        val_ir_stripe_noise_slight_path_list,
        val_ir_stripe_noise_moderate_path_list,
        val_ir_stripe_noise_average_path_list,
        val_ir_stripe_noise_serious_path_list
    ]

    return (
        vi_noise_train_path_list, vi_noise_val_path_list,
        over_exposure_train_path_list, over_exposure_val_path_list,
        ir_low_contrast_train_path_list, ir_low_contrast_val_path_list,
        ir_noise_train_path_list, ir_noise_val_path_list,
        ir_stripe_noise_train_path_list, ir_stripe_noise_val_path_list
    )
