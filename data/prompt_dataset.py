from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random

class PromptDataSet(Dataset):
    def __init__(self, train_vi_noise_path_list, val_vi_noise_path_list, train_over_exposure_path_list, val_over_exposure_path_list,
                 train_ir_low_contrast_path_list, val_ir_low_contrast_path_list, train_ir_noise_path_list, val_ir_noise_path_list, train_ir_stripe_noise_path_list, val_ir_stripe_noise_path_list,
                 train_vi_blur_path_list, val_vi_blur_path_list, train_vi_low_light_path_list, val_vi_low_light_path_list, train_vi_rain_path_list, val_vi_rain_path_list, train_vi_haze_path_list, val_vi_haze_path_list,
                 train_vi_haze_low_path_list, val_vi_haze_low_path_list, train_vi_noise_low_path_list, val_vi_noise_low_path_list, train_vi_rain_haze_path_list, val_vi_rain_haze_path_list,train_llsn_path_list,val_llsn_path_list,
                 train_oelc_path_list,val_oelc_path_list,train_rhrn_path_list,val_rhrn_path_list,
                 phase="train", transform=None):
        self.phase = phase
        if phase == "train":
            self.paths = {
                'vi_blur_slight_A': train_vi_blur_path_list[0][0],
                'vi_blur_slight_B': train_vi_blur_path_list[0][1],
                'vi_blur_moderate_A': train_vi_blur_path_list[1][0],
                'vi_blur_moderate_B': train_vi_blur_path_list[1][1],
                'vi_blur_average_A': train_vi_blur_path_list[2][0],
                'vi_blur_average_B': train_vi_blur_path_list[2][1],
                'vi_blur_extreme_A': train_vi_blur_path_list[3][0],
                'vi_blur_extreme_B': train_vi_blur_path_list[3][1],

                'vi_low_light_slight_A': train_vi_low_light_path_list[0][0],
                'vi_low_light_slight_B': train_vi_low_light_path_list[0][1],
                'vi_low_light_moderate_A': train_vi_low_light_path_list[1][0],
                'vi_low_light_moderate_B': train_vi_low_light_path_list[1][1],
                'vi_low_light_average_A': train_vi_low_light_path_list[2][0],
                'vi_low_light_average_B': train_vi_low_light_path_list[2][1],
                'vi_low_light_extreme_A': train_vi_low_light_path_list[3][0],
                'vi_low_light_extreme_B': train_vi_low_light_path_list[3][1],

                'vi_rain_slight_A': train_vi_rain_path_list[0][0],
                'vi_rain_slight_B': train_vi_rain_path_list[0][1],
                'vi_rain_moderate_A': train_vi_rain_path_list[1][0],
                'vi_rain_moderate_B': train_vi_rain_path_list[1][1],
                'vi_rain_average_A': train_vi_rain_path_list[2][0],
                'vi_rain_average_B': train_vi_rain_path_list[2][1],
                'vi_rain_extreme_A': train_vi_rain_path_list[3][0],
                'vi_rain_extreme_B': train_vi_rain_path_list[3][1],

                'vi_haze_slight_A': train_vi_haze_path_list[0][0],
                'vi_haze_slight_B': train_vi_haze_path_list[0][1],
                'vi_haze_moderate_A': train_vi_haze_path_list[1][0],
                'vi_haze_moderate_B': train_vi_haze_path_list[1][1],
                'vi_haze_average_A': train_vi_haze_path_list[2][0],
                'vi_haze_average_B': train_vi_haze_path_list[2][1],
                'vi_haze_extreme_A': train_vi_haze_path_list[3][0],
                'vi_haze_extreme_B': train_vi_haze_path_list[3][1],

                'vi_haze_low_A': train_vi_haze_low_path_list[0],
                'vi_haze_low_B': train_vi_haze_low_path_list[1],

                'vi_noise_low_A': train_vi_noise_low_path_list[0],
                'vi_noise_low_B': train_vi_noise_low_path_list[1],

                'vi_rain_haze_A': train_vi_rain_haze_path_list[0],
                'vi_rain_haze_B': train_vi_rain_haze_path_list[1],
                
                'vi_llsn_A': train_llsn_path_list[0],
                'vi_llsn_B': train_llsn_path_list[1],

                'vi_oelc_A': train_oelc_path_list[0],
                'vi_oelc_B': train_oelc_path_list[1],

                'vi_rhrn_A': train_rhrn_path_list[0],
                'vi_rhrn_B': train_rhrn_path_list[1],

                'vi_noise_slight_A': train_vi_noise_path_list[0][0],
                'vi_noise_slight_B': train_vi_noise_path_list[0][1],
                'vi_noise_moderate_A': train_vi_noise_path_list[1][0],
                'vi_noise_moderate_B': train_vi_noise_path_list[1][1],
                'vi_noise_average_A': train_vi_noise_path_list[2][0],
                'vi_noise_average_B': train_vi_noise_path_list[2][1],
                'vi_noise_extreme_A': train_vi_noise_path_list[3][0],
                'vi_noise_extreme_B': train_vi_noise_path_list[3][1],

                'over_exposure_slight_A': train_over_exposure_path_list[0][0],
                'over_exposure_slight_B': train_over_exposure_path_list[0][1],
                'over_exposure_moderate_A': train_over_exposure_path_list[1][0],
                'over_exposure_moderate_B': train_over_exposure_path_list[1][1],
                'over_exposure_average_A': train_over_exposure_path_list[2][0],
                'over_exposure_average_B': train_over_exposure_path_list[2][1],
                'over_exposure_extreme_A': train_over_exposure_path_list[3][0],
                'over_exposure_extreme_B': train_over_exposure_path_list[3][1],
                

                'ir_low_contrast_slight_A': train_ir_low_contrast_path_list[0][0],
                'ir_low_contrast_slight_B': train_ir_low_contrast_path_list[0][1],
                'ir_low_contrast_moderate_A': train_ir_low_contrast_path_list[1][0],
                'ir_low_contrast_moderate_B': train_ir_low_contrast_path_list[1][1],
                'ir_low_contrast_average_A': train_ir_low_contrast_path_list[2][0],
                'ir_low_contrast_average_B': train_ir_low_contrast_path_list[2][1],
                'ir_low_contrast_extreme_A': train_ir_low_contrast_path_list[3][0],
                'ir_low_contrast_extreme_B': train_ir_low_contrast_path_list[3][1],


                'ir_noise_slight_A': train_ir_noise_path_list[0][0],
                'ir_noise_slight_B': train_ir_noise_path_list[0][1],
                'ir_noise_moderate_A': train_ir_noise_path_list[1][0],
                'ir_noise_moderate_B': train_ir_noise_path_list[1][1],
                'ir_noise_average_A': train_ir_noise_path_list[2][0],
                'ir_noise_average_B': train_ir_noise_path_list[2][1],
                'ir_noise_extreme_A': train_ir_noise_path_list[3][0],
                'ir_noise_extreme_B': train_ir_noise_path_list[3][1],

                'ir_stripe_noise_slight_A': train_ir_stripe_noise_path_list[0][0],
                'ir_stripe_noise_slight_B': train_ir_stripe_noise_path_list[0][1],
                'ir_stripe_noise_moderate_A': train_ir_stripe_noise_path_list[1][0],
                'ir_stripe_noise_moderate_B': train_ir_stripe_noise_path_list[1][1],
                'ir_stripe_noise_average_A': train_ir_stripe_noise_path_list[2][0],
                'ir_stripe_noise_average_B': train_ir_stripe_noise_path_list[2][1],
                'ir_stripe_noise_extreme_A': train_ir_stripe_noise_path_list[3][0],
                'ir_stripe_noise_extreme_B': train_ir_stripe_noise_path_list[3][1],
            }
            self.paths_gt = {
                'vi_blur_slight_A_gt': train_vi_blur_path_list[0][2],
                'vi_blur_slight_B_gt': train_vi_blur_path_list[0][3],
                'vi_blur_moderate_A_gt': train_vi_blur_path_list[1][2],
                'vi_blur_moderate_B_gt': train_vi_blur_path_list[1][3],
                'vi_blur_average_A_gt': train_vi_blur_path_list[2][2],
                'vi_blur_average_B_gt': train_vi_blur_path_list[2][3],
                'vi_blur_extreme_A_gt': train_vi_blur_path_list[3][2],
                'vi_blur_extreme_B_gt': train_vi_blur_path_list[3][3],

                'vi_low_light_slight_A_gt': train_vi_low_light_path_list[0][2],
                'vi_low_light_slight_B_gt': train_vi_low_light_path_list[0][3],
                'vi_low_light_moderate_A_gt': train_vi_low_light_path_list[1][2],
                'vi_low_light_moderate_B_gt': train_vi_low_light_path_list[1][3],
                'vi_low_light_average_A_gt': train_vi_low_light_path_list[2][2],
                'vi_low_light_average_B_gt': train_vi_low_light_path_list[2][3],
                'vi_low_light_extreme_A_gt': train_vi_low_light_path_list[3][2],
                'vi_low_light_extreme_B_gt': train_vi_low_light_path_list[3][3],

                'vi_rain_slight_A_gt': train_vi_rain_path_list[0][2],
                'vi_rain_slight_B_gt': train_vi_rain_path_list[0][3],
                'vi_rain_moderate_A_gt': train_vi_rain_path_list[1][2],
                'vi_rain_moderate_B_gt': train_vi_rain_path_list[1][3],
                'vi_rain_average_A_gt': train_vi_rain_path_list[2][2],
                'vi_rain_average_B_gt': train_vi_rain_path_list[2][3],
                'vi_rain_extreme_A_gt': train_vi_rain_path_list[3][2],
                'vi_rain_extreme_B_gt': train_vi_rain_path_list[3][3],

                'vi_haze_slight_A_gt': train_vi_haze_path_list[0][2],
                'vi_haze_slight_B_gt': train_vi_haze_path_list[0][3],
                'vi_haze_moderate_A_gt': train_vi_haze_path_list[1][2],
                'vi_haze_moderate_B_gt': train_vi_haze_path_list[1][3],
                'vi_haze_average_A_gt': train_vi_haze_path_list[2][2],
                'vi_haze_average_B_gt': train_vi_haze_path_list[2][3],
                'vi_haze_extreme_A_gt': train_vi_haze_path_list[3][2],
                'vi_haze_extreme_B_gt': train_vi_haze_path_list[3][3],

                'vi_haze_low_A_gt': train_vi_haze_low_path_list[2],
                'vi_haze_low_B_gt': train_vi_haze_low_path_list[3],

                'vi_noise_low_A_gt': train_vi_noise_low_path_list[2],
                'vi_noise_low_B_gt': train_vi_noise_low_path_list[3],

                'vi_rain_haze_A_gt': train_vi_rain_haze_path_list[2],
                'vi_rain_haze_B_gt': train_vi_rain_haze_path_list[3],
                
                'vi_llsn_A_gt': train_llsn_path_list[2],
                'vi_llsn_B_gt': train_llsn_path_list[3],

                'vi_oelc_A_gt': train_oelc_path_list[2],
                'vi_oelc_B_gt': train_oelc_path_list[3],

                'vi_rhrn_A_gt': train_rhrn_path_list[2],
                'vi_rhrn_B_gt': train_rhrn_path_list[3],

                'vi_noise_slight_A_gt': train_vi_noise_path_list[0][2],
                'vi_noise_slight_B_gt': train_vi_noise_path_list[0][3],
                'vi_noise_moderate_A_gt': train_vi_noise_path_list[1][2],
                'vi_noise_moderate_B_gt': train_vi_noise_path_list[1][3],
                'vi_noise_average_A_gt': train_vi_noise_path_list[2][2],
                'vi_noise_average_B_gt': train_vi_noise_path_list[2][3],
                'vi_noise_extreme_A_gt': train_vi_noise_path_list[3][2],
                'vi_noise_extreme_B_gt': train_vi_noise_path_list[3][3],

                'over_exposure_slight_A_gt': train_over_exposure_path_list[0][2],
                'over_exposure_slight_B_gt': train_over_exposure_path_list[0][3],
                'over_exposure_moderate_A_gt': train_over_exposure_path_list[1][2],
                'over_exposure_moderate_B_gt': train_over_exposure_path_list[1][3],
                'over_exposure_average_A_gt': train_over_exposure_path_list[2][2],
                'over_exposure_average_B_gt': train_over_exposure_path_list[2][3],
                'over_exposure_extreme_A_gt': train_over_exposure_path_list[3][2],
                'over_exposure_extreme_B_gt': train_over_exposure_path_list[3][3],
                

                'ir_low_contrast_slight_A_gt': train_ir_low_contrast_path_list[0][2],
                'ir_low_contrast_slight_B_gt': train_ir_low_contrast_path_list[0][3],
                'ir_low_contrast_moderate_A_gt': train_ir_low_contrast_path_list[1][2],
                'ir_low_contrast_moderate_B_gt': train_ir_low_contrast_path_list[1][3],
                'ir_low_contrast_average_A_gt': train_ir_low_contrast_path_list[2][2],
                'ir_low_contrast_average_B_gt': train_ir_low_contrast_path_list[2][3],
                'ir_low_contrast_extreme_A_gt': train_ir_low_contrast_path_list[3][2],
                'ir_low_contrast_extreme_B_gt': train_ir_low_contrast_path_list[3][3],


                'ir_noise_slight_A_gt': train_ir_noise_path_list[0][2],
                'ir_noise_slight_B_gt': train_ir_noise_path_list[0][3],
                'ir_noise_moderate_A_gt': train_ir_noise_path_list[1][2],
                'ir_noise_moderate_B_gt': train_ir_noise_path_list[1][3],
                'ir_noise_average_A_gt': train_ir_noise_path_list[2][2],
                'ir_noise_average_B_gt': train_ir_noise_path_list[2][3],
                'ir_noise_extreme_A_gt': train_ir_noise_path_list[3][2],
                'ir_noise_extreme_B_gt': train_ir_noise_path_list[3][3],

                'ir_stripe_noise_slight_A_gt': train_ir_stripe_noise_path_list[0][2],
                'ir_stripe_noise_slight_B_gt': train_ir_stripe_noise_path_list[0][3],
                'ir_stripe_noise_moderate_A_gt': train_ir_stripe_noise_path_list[1][2],
                'ir_stripe_noise_moderate_B_gt': train_ir_stripe_noise_path_list[1][3],
                'ir_stripe_noise_average_A_gt': train_ir_stripe_noise_path_list[2][2],
                'ir_stripe_noise_average_B_gt': train_ir_stripe_noise_path_list[2][3],
                'ir_stripe_noise_extreme_A_gt': train_ir_stripe_noise_path_list[3][2],
                'ir_stripe_noise_extreme_B_gt': train_ir_stripe_noise_path_list[3][3],
            }
        else:
            self.paths = {
                'vi_blur_slight_A': val_vi_blur_path_list[0][0],
                'vi_blur_slight_B': val_vi_blur_path_list[0][1],
                'vi_blur_moderate_A': val_vi_blur_path_list[1][0],
                'vi_blur_moderate_B': val_vi_blur_path_list[1][1],
                'vi_blur_average_A': val_vi_blur_path_list[2][0],
                'vi_blur_average_B': val_vi_blur_path_list[2][1],
                'vi_blur_extreme_A': val_vi_blur_path_list[3][0],
                'vi_blur_extreme_B': val_vi_blur_path_list[3][1],

                'vi_low_light_slight_A': val_vi_low_light_path_list[0][0],
                'vi_low_light_slight_B': val_vi_low_light_path_list[0][1],
                'vi_low_light_moderate_A': val_vi_low_light_path_list[1][0],
                'vi_low_light_moderate_B': val_vi_low_light_path_list[1][1],
                'vi_low_light_average_A': val_vi_low_light_path_list[2][0],
                'vi_low_light_average_B': val_vi_low_light_path_list[2][1],
                'vi_low_light_extreme_A': val_vi_low_light_path_list[3][0],
                'vi_low_light_extreme_B': val_vi_low_light_path_list[3][1],

                'vi_rain_slight_A': val_vi_rain_path_list[0][0],
                'vi_rain_slight_B': val_vi_rain_path_list[0][1],
                'vi_rain_moderate_A': val_vi_rain_path_list[1][0],
                'vi_rain_moderate_B': val_vi_rain_path_list[1][1],
                'vi_rain_average_A': val_vi_rain_path_list[2][0],
                'vi_rain_average_B': val_vi_rain_path_list[2][1],
                'vi_rain_extreme_A': val_vi_rain_path_list[3][0],
                'vi_rain_extreme_B': val_vi_rain_path_list[3][1],

                'vi_haze_slight_A': val_vi_haze_path_list[0][0],
                'vi_haze_slight_B': val_vi_haze_path_list[0][1],
                'vi_haze_moderate_A': val_vi_haze_path_list[1][0],
                'vi_haze_moderate_B': val_vi_haze_path_list[1][1],
                'vi_haze_average_A': val_vi_haze_path_list[2][0],
                'vi_haze_average_B': val_vi_haze_path_list[2][1],
                'vi_haze_extreme_A': val_vi_haze_path_list[3][0],
                'vi_haze_extreme_B': val_vi_haze_path_list[3][1],

                'vi_haze_low_A': val_vi_haze_low_path_list[0],
                'vi_haze_low_B': val_vi_haze_low_path_list[1],

                'vi_noise_low_A': val_vi_noise_low_path_list[0],
                'vi_noise_low_B': val_vi_noise_low_path_list[1],

                'vi_rain_haze_A': val_vi_rain_haze_path_list[0],
                'vi_rain_haze_B': val_vi_rain_haze_path_list[1],

                'vi_llsn_A': val_llsn_path_list[0],
                'vi_llsn_B': val_llsn_path_list[1],

                'vi_oelc_A': val_oelc_path_list[0],
                'vi_oelc_B': val_oelc_path_list[1],

                'vi_rhrn_A': val_rhrn_path_list[0],
                'vi_rhrn_B': val_rhrn_path_list[1],
                
                'vi_noise_slight_A': val_vi_noise_path_list[0][0],
                'vi_noise_slight_B': val_vi_noise_path_list[0][1],
                'vi_noise_moderate_A': val_vi_noise_path_list[1][0],
                'vi_noise_moderate_B': val_vi_noise_path_list[1][1],
                'vi_noise_average_A': val_vi_noise_path_list[2][0],
                'vi_noise_average_B': val_vi_noise_path_list[2][1],
                'vi_noise_extreme_A': val_vi_noise_path_list[3][0],
                'vi_noise_extreme_B': val_vi_noise_path_list[3][1],

                'over_exposure_slight_A': val_over_exposure_path_list[0][0],
                'over_exposure_slight_B': val_over_exposure_path_list[0][1],
                'over_exposure_moderate_A': val_over_exposure_path_list[1][0],
                'over_exposure_moderate_B': val_over_exposure_path_list[1][1],
                'over_exposure_average_A': val_over_exposure_path_list[2][0],
                'over_exposure_average_B': val_over_exposure_path_list[2][1],
                'over_exposure_extreme_A': val_over_exposure_path_list[3][0],
                'over_exposure_extreme_B': val_over_exposure_path_list[3][1],
                

                'ir_low_contrast_slight_A': val_ir_low_contrast_path_list[0][0],
                'ir_low_contrast_slight_B': val_ir_low_contrast_path_list[0][1],
                'ir_low_contrast_moderate_A': val_ir_low_contrast_path_list[1][0],
                'ir_low_contrast_moderate_B': val_ir_low_contrast_path_list[1][1],
                'ir_low_contrast_average_A': val_ir_low_contrast_path_list[2][0],
                'ir_low_contrast_average_B': val_ir_low_contrast_path_list[2][1],
                'ir_low_contrast_extreme_A': val_ir_low_contrast_path_list[3][0],
                'ir_low_contrast_extreme_B': val_ir_low_contrast_path_list[3][1],


                'ir_noise_slight_A': val_ir_noise_path_list[0][0],
                'ir_noise_slight_B': val_ir_noise_path_list[0][1],
                'ir_noise_moderate_A': val_ir_noise_path_list[1][0],
                'ir_noise_moderate_B': val_ir_noise_path_list[1][1],
                'ir_noise_average_A': val_ir_noise_path_list[2][0],
                'ir_noise_average_B': val_ir_noise_path_list[2][1],
                'ir_noise_extreme_A': val_ir_noise_path_list[3][0],
                'ir_noise_extreme_B': val_ir_noise_path_list[3][1],

                'ir_stripe_noise_slight_A': val_ir_stripe_noise_path_list[0][0],
                'ir_stripe_noise_slight_B': val_ir_stripe_noise_path_list[0][1],
                'ir_stripe_noise_moderate_A': val_ir_stripe_noise_path_list[1][0],
                'ir_stripe_noise_moderate_B': val_ir_stripe_noise_path_list[1][1],
                'ir_stripe_noise_average_A': val_ir_stripe_noise_path_list[2][0],
                'ir_stripe_noise_average_B': val_ir_stripe_noise_path_list[2][1],
                'ir_stripe_noise_extreme_A': val_ir_stripe_noise_path_list[3][0],
                'ir_stripe_noise_extreme_B': val_ir_stripe_noise_path_list[3][1],
            }
            self.paths_gt = {
                'vi_blur_slight_A_gt': val_vi_blur_path_list[0][0],
                'vi_blur_slight_B_gt': val_vi_blur_path_list[0][1],
                'vi_blur_moderate_A_gt': val_vi_blur_path_list[1][0],
                'vi_blur_moderate_B_gt': val_vi_blur_path_list[1][1],
                'vi_blur_average_A_gt': val_vi_blur_path_list[2][0],
                'vi_blur_average_B_gt': val_vi_blur_path_list[2][1],
                'vi_blur_extreme_A_gt': val_vi_blur_path_list[3][0],
                'vi_blur_extreme_B_gt': val_vi_blur_path_list[3][1],


                'vi_low_light_slight_A_gt': val_vi_low_light_path_list[0][0],
                'vi_low_light_slight_B_gt': val_vi_low_light_path_list[0][1],
                'vi_low_light_moderate_A_gt': val_vi_low_light_path_list[1][0],
                'vi_low_light_moderate_B_gt': val_vi_low_light_path_list[1][1],
                'vi_low_light_average_A_gt': val_vi_low_light_path_list[2][0],
                'vi_low_light_average_B_gt': val_vi_low_light_path_list[2][1],
                'vi_low_light_extreme_A_gt': val_vi_low_light_path_list[3][0],
                'vi_low_light_extreme_B_gt': val_vi_low_light_path_list[3][1],

                'vi_rain_slight_A_gt': val_vi_rain_path_list[0][0],
                'vi_rain_slight_B_gt': val_vi_rain_path_list[0][1],
                'vi_rain_moderate_A_gt': val_vi_rain_path_list[1][0],
                'vi_rain_moderate_B_gt': val_vi_rain_path_list[1][1],
                'vi_rain_average_A_gt': val_vi_rain_path_list[2][0],
                'vi_rain_average_B_gt': val_vi_rain_path_list[2][1],
                'vi_rain_extreme_A_gt': val_vi_rain_path_list[3][0],
                'vi_rain_extreme_B_gt': val_vi_rain_path_list[3][1],

                'vi_haze_slight_A_gt': val_vi_haze_path_list[0][0],
                'vi_haze_slight_B_gt': val_vi_haze_path_list[0][1],
                'vi_haze_moderate_A_gt': val_vi_haze_path_list[1][0],
                'vi_haze_moderate_B_gt': val_vi_haze_path_list[1][1],
                'vi_haze_average_A_gt': val_vi_haze_path_list[2][0],
                'vi_haze_average_B_gt': val_vi_haze_path_list[2][1],
                'vi_haze_extreme_A_gt': val_vi_haze_path_list[3][0],
                'vi_haze_extreme_B_gt': val_vi_haze_path_list[3][1],

                'vi_haze_low_A_gt': val_vi_haze_low_path_list[0],
                'vi_haze_low_B_gt': val_vi_haze_low_path_list[1],
                
                'vi_noise_low_A_gt': val_vi_noise_low_path_list[0],
                'vi_noise_low_B_gt': val_vi_noise_low_path_list[1],

                'vi_rain_haze_A_gt': val_vi_rain_haze_path_list[0],
                'vi_rain_haze_B_gt': val_vi_rain_haze_path_list[1],
                
                'vi_llsn_A_gt': val_llsn_path_list[0],
                'vi_llsn_B_gt': val_llsn_path_list[1],

                'vi_oelc_A_gt': val_oelc_path_list[0],
                'vi_oelc_B_gt': val_oelc_path_list[1],

                'vi_rhrn_A_gt': val_rhrn_path_list[0],
                'vi_rhrn_B_gt': val_rhrn_path_list[1],

                'vi_noise_slight_A_gt': val_vi_noise_path_list[0][0],
                'vi_noise_slight_B_gt': val_vi_noise_path_list[0][1],
                'vi_noise_moderate_A_gt': val_vi_noise_path_list[1][0],
                'vi_noise_moderate_B_gt': val_vi_noise_path_list[1][1],
                'vi_noise_average_A_gt': val_vi_noise_path_list[2][0],
                'vi_noise_average_B_gt': val_vi_noise_path_list[2][1],
                'vi_noise_extreme_A_gt': val_vi_noise_path_list[3][0],
                'vi_noise_extreme_B_gt': val_vi_noise_path_list[3][1],

                'over_exposure_slight_A_gt': val_over_exposure_path_list[0][0],
                'over_exposure_slight_B_gt': val_over_exposure_path_list[0][1],
                'over_exposure_moderate_A_gt': val_over_exposure_path_list[1][0],
                'over_exposure_moderate_B_gt': val_over_exposure_path_list[1][1],
                'over_exposure_average_A_gt': val_over_exposure_path_list[2][0],
                'over_exposure_average_B_gt': val_over_exposure_path_list[2][1],
                'over_exposure_extreme_A_gt': val_over_exposure_path_list[3][0],
                'over_exposure_extreme_B_gt': val_over_exposure_path_list[3][1],
                

                'ir_low_contrast_slight_A_gt': val_ir_low_contrast_path_list[0][0],
                'ir_low_contrast_slight_B_gt': val_ir_low_contrast_path_list[0][1],
                'ir_low_contrast_moderate_A_gt': val_ir_low_contrast_path_list[1][0],
                'ir_low_contrast_moderate_B_gt': val_ir_low_contrast_path_list[1][1],
                'ir_low_contrast_average_A_gt': val_ir_low_contrast_path_list[2][0],
                'ir_low_contrast_average_B_gt': val_ir_low_contrast_path_list[2][1],
                'ir_low_contrast_extreme_A_gt': val_ir_low_contrast_path_list[3][0],
                'ir_low_contrast_extreme_B_gt': val_ir_low_contrast_path_list[3][1],


                'ir_noise_slight_A_gt': val_ir_noise_path_list[0][0],
                'ir_noise_slight_B_gt': val_ir_noise_path_list[0][1],
                'ir_noise_moderate_A_gt': val_ir_noise_path_list[1][0],
                'ir_noise_moderate_B_gt': val_ir_noise_path_list[1][1],
                'ir_noise_average_A_gt': val_ir_noise_path_list[2][0],
                'ir_noise_average_B_gt': val_ir_noise_path_list[2][1],
                'ir_noise_extreme_A_gt': val_ir_noise_path_list[3][0],
                'ir_noise_extreme_B_gt': val_ir_noise_path_list[3][1],

                'ir_stripe_noise_slight_A_gt': val_ir_stripe_noise_path_list[0][0],
                'ir_stripe_noise_slight_B_gt': val_ir_stripe_noise_path_list[0][1],
                'ir_stripe_noise_moderate_A_gt': val_ir_stripe_noise_path_list[1][0],
                'ir_stripe_noise_moderate_B_gt': val_ir_stripe_noise_path_list[1][1],
                'ir_stripe_noise_average_A_gt': val_ir_stripe_noise_path_list[2][0],
                'ir_stripe_noise_average_B_gt': val_ir_stripe_noise_path_list[2][1],
                'ir_stripe_noise_extreme_A_gt': val_ir_stripe_noise_path_list[3][0],
                'ir_stripe_noise_extreme_B_gt': val_ir_stripe_noise_path_list[3][1],
            }
        self.transform = transform

        # Create a list to hold all sample indices grouped by class
        self.class_indices = {}
        for class_key, paths in self.paths.items():
            self.class_indices[class_key] = list(range(len(paths)))
        pass

    def __len__(self):
        if self.phase == "train":
            return sum(len(paths) for paths in self.paths.values())
        else:
            # Return the part number of images in val all classes and subsets
            #return sum(len(paths) for paths in self.paths.values()) // 4
            return 80

    def __getitem__(self, item):
        # Randomly select a class, use the random sampling (equal to sequential sampling when the number of sampling is large)
        class_key = random.choice(list(self.paths.keys()))

        # Randomly select an index for the chosen class
        class_indices = self.class_indices[class_key]
        item_index = random.randint(0, len(class_indices) - 1)
        image_index = class_indices[item_index]

        # Load the A and B images based on the class and index
        image_A_path = self.paths[class_key[:-2] + '_A'][image_index]
        image_B_path = self.paths[class_key[:-2] + '_B'][image_index]

        image_A_gt_path = self.paths_gt[class_key[:-2] + '_A_gt'][image_index]
        image_B_gt_path = self.paths_gt[class_key[:-2] + '_B_gt'][image_index]

        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')
        image_A_gt = Image.open(image_A_gt_path).convert(mode='RGB')
        image_B_gt = Image.open(image_B_gt_path).convert(mode='RGB')

        image_full = image_A
        # Apply any specified transformations
        if self.transform is not None:
            image_A, image_B, image_A_gt, image_B_gt, image_full = self.transform(image_A, image_B, image_A_gt, image_B_gt, image_full)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, image_A_gt, image_B_gt, image_full, class_key[:-2], name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        images_A_gt = torch.stack(images_A_gt, dim=0)
        images_B_gt = torch.stack(images_B_gt, dim=0)
        images_full = torch.stack(images_full, dim=0)
        return images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name