import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import clip
from data.prompt_dataset import PromptDataSet


from model.ControlFusion import ControlFusion as create_model
from scripts.utils import read_data, train_one_epoch, evaluate, create_lr_scheduler, merge_and_print_paths
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import transforms as T

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/ControlFusion_train_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_val_loss = 1e5
    start_epoch = 0

    print("Loading IVF Fusion and noise slight Task!")
    if args.vi_noise_slight_path is not None:
        train_vi_noise_slight_path_list, val_vi_noise_slight_path_list = read_data(args.vi_noise_slight_path)
    else:
        train_vi_noise_slight_path_list = val_vi_noise_slight_path_list = None

    print("Loading IVF Fusion and noise moderate Task!")
    if args.vi_noise_moderate_path is not None:
        train_vi_noise_moderate_path_list, val_vi_noise_moderate_path_list = read_data(args.vi_noise_moderate_path)
    else:
        train_vi_noise_moderate_path_list = val_vi_noise_moderate_path_list = None

    print("Loading IVF Fusion and noise average Task!")
    if args.vi_noise_average_path is not None:
        train_vi_noise_average_path_list, val_vi_noise_average_path_list = read_data(args.vi_noise_average_path)
    else:
        train_vi_noise_average_path_list = val_vi_noise_average_path_list = None

    print("Loading IVF Fusion and noise serious Task!")
    if args.vi_noise_serious_path is not None:
        train_vi_noise_serious_path_list, val_vi_noise_serious_path_list = read_data(args.vi_noise_serious_path)
    else:
        train_vi_noise_serious_path_list = val_vi_noise_serious_path_list = None


    print("Loading IVF Fusion and Over-Exposure slight Task!")
    if args.over_exposure_slight_path is not None:
        train_over_exposure_slight_path_list, val_over_exposure_slight_path_list = read_data(args.over_exposure_slight_path)
    else:
        train_over_exposure_slight_path_list = val_over_exposure_slight_path_list = None

    print("Loading IVF Fusion and Over-Exposure moderate Task!")
    if args.over_exposure_moderate_path is not None:
        train_over_exposure_moderate_path_list, val_over_exposure_moderate_path_list = read_data(args.over_exposure_moderate_path)
    else:
        train_over_exposure_moderate_path_list = val_over_exposure_moderate_path_list = None

    print("Loading IVF Fusion and Over-Exposure average Task!")
    if args.over_exposure_average_path is not None:
        train_over_exposure_average_path_list, val_over_exposure_average_path_list = read_data(args.over_exposure_average_path)
    else:
        train_over_exposure_average_path_list = val_over_exposure_average_path_list = None

    print("Loading IVF Fusion and Over-Exposure serious Task!")
    if args.over_exposure_serious_path is not None:
        train_over_exposure_serious_path_list, val_over_exposure_serious_path_list = read_data(args.over_exposure_serious_path)
    else:
        train_over_exposure_serious_path_list = val_over_exposure_serious_path_list = None

    print("Loading IVF Fusion and Blur slight Task!")
    if args.vi_blur_slight_path is not None:
        train_vi_blur_slight_path_list, val_vi_blur_slight_path_list = read_data(args.vi_blur_slight_path)
    else:
        train_vi_blur_slight_path_list = val_vi_blur_slight_path_list = None

    print("Loading IVF Fusion and Blur moderate Task!")
    if args.vi_blur_moderate_path is not None:
        train_vi_blur_moderate_path_list, val_vi_blur_moderate_path_list = read_data(args.vi_blur_moderate_path)
    else:
        train_vi_blur_moderate_path_list = val_vi_blur_moderate_path_list = None
    
    print("Loading IVF Fusion and Blur average Task!")
    if args.vi_blur_average_path is not None:
        train_vi_blur_average_path_list, val_vi_blur_average_path_list = read_data(args.vi_blur_average_path)
    else:
        train_vi_blur_average_path_list = val_vi_blur_average_path_list = None

    print("Loading IVF Fusion and Blur serious Task!")
    if args.vi_blur_serious_path is not None:
        train_vi_blur_serious_path_list, val_vi_blur_serious_path_list = read_data(args.vi_blur_serious_path)
    else:
        train_vi_blur_serious_path_list = val_vi_blur_serious_path_list = None

    print("Loading IVF Fusion and Haze slight Task!")
    if args.vi_haze_slight_path is not None:
        train_vi_haze_slight_path_list, val_vi_haze_slight_path_list = read_data(args.vi_haze_slight_path)
    else:
        train_vi_haze_slight_path_list = val_vi_haze_slight_path_list = None
    
    print("Loading IVF Fusion and Haze moderate Task!")
    if args.vi_haze_moderate_path is not None:
        train_vi_haze_moderate_path_list, val_vi_haze_moderate_path_list = read_data(args.vi_haze_moderate_path)
    else:
        train_vi_haze_moderate_path_list = val_vi_haze_moderate_path_list = None
    
    print("Loading IVF Fusion and Haze average Task!")
    if args.vi_haze_average_path is not None:
        train_vi_haze_average_path_list, val_vi_haze_average_path_list = read_data(args.vi_haze_average_path)
    else:
        train_vi_haze_average_path_list = val_vi_haze_average_path_list = None

    print("Loading IVF Fusion and Haze serious Task!")
    if args.vi_haze_serious_path is not None:
        train_vi_haze_serious_path_list, val_vi_haze_serious_path_list = read_data(args.vi_haze_serious_path)
    else:
        train_vi_haze_serious_path_list = val_vi_haze_serious_path_list = None
    
    print("Loading IVF Fusion and Rain slight Task!")
    if args.vi_rain_slight_path is not None:
        train_vi_rain_slight_path_list, val_vi_rain_slight_path_list = read_data(args.vi_rain_slight_path)
    else:
        train_vi_rain_slight_path_list = val_vi_rain_slight_path_list = None
    
    print("Loading IVF Fusion and Rain moderate Task!")
    if args.vi_rain_moderate_path is not None:
        train_vi_rain_moderate_path_list, val_vi_rain_moderate_path_list = read_data(args.vi_rain_moderate_path)
    else:
        train_vi_rain_moderate_path_list = val_vi_rain_moderate_path_list = None
    
    print("Loading IVF Fusion and Rain average Task!")
    if args.vi_rain_average_path is not None:
        train_vi_rain_average_path_list, val_vi_rain_average_path_list = read_data(args.vi_rain_average_path)
    else:
        train_vi_rain_average_path_list = val_vi_rain_average_path_list = None

    print("Loading IVF Fusion and Rain serious Task!")
    if args.vi_rain_serious_path is not None:
        train_vi_rain_serious_path_list, val_vi_rain_serious_path_list = read_data(args.vi_rain_serious_path)
    else:
        train_vi_rain_serious_path_list = val_vi_rain_serious_path_list = None
    
    print("Loading IVF Fusion and Low Light slight Task!")
    if args.vi_low_light_slight_path is not None:
        train_vi_low_light_slight_path_list, val_vi_low_light_slight_path_list = read_data(args.vi_low_light_slight_path)
    else:
        train_vi_low_light_slight_path_list = val_vi_low_light_slight_path_list = None

    print("Loading IVF Fusion and Low Light moderate Task!")
    if args.vi_low_light_moderate_path is not None:
        train_vi_low_light_moderate_path_list, val_vi_low_light_moderate_path_list = read_data(args.vi_low_light_moderate_path)
    else:
        train_vi_low_light_moderate_path_list = val_vi_low_light_moderate_path_list = None
    
    print("Loading IVF Fusion and Low Light average Task!")
    if args.vi_low_light_average_path is not None:
        train_vi_low_light_average_path_list, val_vi_low_light_average_path_list = read_data(args.vi_low_light_average_path)
    else:
        train_vi_low_light_average_path_list = val_vi_low_light_average_path_list = None
    
    print("Loading IVF Fusion and Low Light serious Task!")
    if args.vi_low_light_serious_path is not None:
        train_vi_low_light_serious_path_list, val_vi_low_light_serious_path_list = read_data(args.vi_low_light_serious_path)
    else:
        train_vi_low_light_serious_path_list = val_vi_low_light_serious_path_list = None

    print("Loading IVF Fusion and Haze Low Task!")
    if args.vi_haze_low_path is not None:
        train_vi_haze_low_path_list, val_vi_haze_low_path_list = read_data(args.vi_haze_low_path)
    else:
        train_vi_haze_low_path_list = val_vi_haze_low_path_list = None
        
    print("Loading IVF Fusion and llsn Task!")
    if args.llsn is not None:
        train_llsn_path_list, val_llsn_path_list = read_data(args.llsn)
    else:
        train_llsn_path_list = val_llsn_path_list = None
    
    print("Loading IVF Fusion and oelc Task!")
    if args.oelc is not None:
        train_oelc_path_list, val_oelc_path_list = read_data(args.oelc)
    else:
        train_oelc_path_list = val_oelc_path_list = None
    
    print("Loading IVF Fusion and rhrn Task!")
    if args.rhrn is not None:
        train_rhrn_path_list, val_rhrn_path_list = read_data(args.rhrn)
    else:
        train_rhrn_path_list = val_rhrn_path_list = None

    print("Loading IVF Fusion and Noise Low Task!")
    if args.vi_noise_low_path is not None:
        train_vi_noise_low_path_list, val_vi_noise_low_path_list = read_data(args.vi_noise_low_path)
    else:
        train_vi_noise_low_path_list = val_vi_noise_low_path_list = None
    
    print("Loading IVF Fusion and Rain Haze Task!")
    if args.vi_rain_haze_path is not None:
        train_vi_rain_haze_path_list, val_vi_rain_haze_path_list = read_data(args.vi_rain_haze_path)
    else:
        train_vi_rain_haze_path_list = val_vi_rain_haze_path_list = None


    print("Loading IVF Fusion and ir_low_contrast slight Task!")
    if args.ir_low_contrast_slight_path is not None:
        train_ir_low_contrast_slight_path_list, val_ir_low_contrast_slight_path_list = read_data(args.ir_low_contrast_slight_path)
    else:
        train_ir_low_contrast_slight_path_list = val_ir_low_contrast_slight_path_list = None

    print("Loading IVF Fusion and ir_low_contrast moderate Task!")
    if args.ir_low_contrast_moderate_path is not None:
        train_ir_low_contrast_moderate_path_list, val_ir_low_contrast_moderate_path_list = read_data(args.ir_low_contrast_moderate_path)
    else:
        train_ir_low_contrast_moderate_path_list = val_ir_low_contrast_moderate_path_list = None

    print("Loading IVF Fusion and ir_low_contrast average Task!")
    if args.ir_low_contrast_average_path is not None:
        train_ir_low_contrast_average_path_list, val_ir_low_contrast_average_path_list = read_data(args.ir_low_contrast_average_path)
    else:
        train_ir_low_contrast_average_path_list = val_ir_low_contrast_average_path_list = None

    print("Loading IVF Fusion and ir_low_contrast serious Task!")
    if args.ir_low_contrast_serious_path is not None:
        train_ir_low_contrast_serious_path_list, val_ir_low_contrast_serious_path_list = read_data(args.ir_low_contrast_serious_path)
    else:
        train_ir_low_contrast_serious_path_list = val_ir_low_contrast_serious_path_list = None



    print("Loading IVF Fusion and ir_noise_slight Task!")
    if args.ir_noise_slight_path is not None:
        train_ir_noise_slight_path_list, val_ir_noise_slight_path_list = read_data(args.ir_noise_slight_path)
    else:
        train_ir_noise_slight_path_list = val_ir_noise_slight_path_list = None

    print("Loading IVF Fusion and ir_noise_moderate Task!")
    if args.ir_noise_moderate_path is not None:
        train_ir_noise_moderate_path_list, val_ir_noise_moderate_path_list = read_data(args.ir_noise_moderate_path)
    else:
        train_ir_noise_moderate_path_list = val_ir_noise_moderate_path_list = None

    print("Loading IVF Fusion and ir_noise_average Task!")
    if args.ir_noise_average_path is not None:
        train_ir_noise_average_path_list, val_ir_noise_average_path_list = read_data(args.ir_noise_average_path)
    else:
        train_ir_noise_average_path_list = val_ir_noise_average_path_list = None

    print("Loading IVF Fusion and ir_noise_serious Task!")
    if args.ir_noise_serious_path is not None:
        train_ir_noise_serious_path_list, val_ir_noise_serious_path_list = read_data(args.ir_noise_serious_path)
    else:
        train_ir_noise_serious_path_list = val_ir_noise_serious_path_list = None

    print("Loading IVF Fusion and ir_stripe_noise_slight Task!")
    if args.ir_stripe_noise_slight_path is not None:
        train_ir_stripe_noise_slight_path_list, val_ir_stripe_noise_slight_path_list = read_data(args.ir_stripe_noise_slight_path)
    else:
        train_ir_stripe_noise_slight_path_list = val_ir_stripe_noise_slight_path_list = None

    print("Loading IVF Fusion and ir_stripe_noise_moderate Task!")
    if args.ir_stripe_noise_moderate_path is not None:
        train_ir_stripe_noise_moderate_path_list, val_ir_stripe_noise_moderate_path_list = read_data(args.ir_stripe_noise_moderate_path)
    else:
        train_ir_stripe_noise_moderate_path_list = val_ir_stripe_noise_moderate_path_list = None

    print("Loading IVF Fusion and ir_stripe_noise_average Task!")
    if args.ir_stripe_noise_average_path is not None:
        train_ir_stripe_noise_average_path_list, val_ir_stripe_noise_average_path_list = read_data(args.ir_stripe_noise_average_path)
    else:
        train_ir_stripe_noise_average_path_list = val_ir_stripe_noise_average_path_list = None

    print("Loading IVF Fusion and ir_stripe_noise_serious Task!")
    if args.ir_stripe_noise_serious_path is not None:
        train_ir_stripe_noise_serious_path_list, val_ir_stripe_noise_serious_path_list = read_data(args.ir_stripe_noise_serious_path)
    else:
        train_ir_stripe_noise_serious_path_list = val_ir_stripe_noise_serious_path_list = None

    # 调用函数并接收返回值
    (
        vi_noise_train, vi_noise_val,
        over_exposure_train, over_exposure_val,
        vi_blur_train, vi_blur_val,
        vi_haze_train, vi_haze_val,
        vi_low_light_train, vi_low_light_val,
        vi_rain_train, vi_rain_val,
        ir_low_contrast_train, ir_low_contrast_val,
        ir_noise_train, ir_noise_val,
        ir_stripe_noise_train, ir_stripe_noise_val
    ) = merge_and_print_paths(
        train_vi_noise_slight_path_list, val_vi_noise_slight_path_list,
        train_vi_noise_moderate_path_list, val_vi_noise_moderate_path_list,
        train_vi_noise_average_path_list, val_vi_noise_average_path_list,
        train_vi_noise_serious_path_list, val_vi_noise_serious_path_list,
        train_over_exposure_slight_path_list, val_over_exposure_slight_path_list,
        train_over_exposure_moderate_path_list, val_over_exposure_moderate_path_list,
        train_over_exposure_average_path_list, val_over_exposure_average_path_list,
        train_over_exposure_serious_path_list, val_over_exposure_serious_path_list,
        train_vi_blur_slight_path_list, val_vi_blur_slight_path_list,
        train_vi_blur_moderate_path_list, val_vi_blur_moderate_path_list,
        train_vi_blur_average_path_list, val_vi_blur_average_path_list,
        train_vi_blur_serious_path_list, val_vi_blur_serious_path_list,
        train_vi_haze_slight_path_list, val_vi_haze_slight_path_list,
        train_vi_haze_moderate_path_list, val_vi_haze_moderate_path_list,
        train_vi_haze_average_path_list, val_vi_haze_average_path_list,
        train_vi_haze_serious_path_list, val_vi_haze_serious_path_list,
        train_vi_low_light_slight_path_list, val_vi_low_light_slight_path_list,
        train_vi_low_light_moderate_path_list, val_vi_low_light_moderate_path_list,
        train_vi_low_light_average_path_list, val_vi_low_light_average_path_list,
        train_vi_low_light_serious_path_list, val_vi_low_light_serious_path_list,
        train_vi_rain_slight_path_list, val_vi_rain_slight_path_list,
        train_vi_rain_moderate_path_list, val_vi_rain_moderate_path_list,
        train_vi_rain_average_path_list, val_vi_rain_average_path_list,
        train_vi_rain_serious_path_list, val_vi_rain_serious_path_list,
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
    )


    data_transform = {
        "train": T.Compose([T.RandomCrop(96),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = PromptDataSet(train_vi_noise_path_list=vi_noise_train,
                                  val_vi_noise_path_list=vi_noise_val,
                                  train_over_exposure_path_list=over_exposure_train,
                                  val_over_exposure_path_list=over_exposure_val,
                                  train_ir_low_contrast_path_list=ir_low_contrast_train,
                                  val_ir_low_contrast_path_list=ir_low_contrast_val,
                                  train_ir_noise_path_list=ir_noise_train,
                                  val_ir_noise_path_list=ir_noise_val,
                                  train_ir_stripe_noise_path_list = ir_stripe_noise_train,
                                  val_ir_stripe_noise_path_list = ir_stripe_noise_val,
                                  train_vi_blur_path_list=vi_blur_train,
                                  val_vi_blur_path_list=vi_blur_val,
                                  train_vi_haze_path_list=vi_haze_train,
                                  val_vi_haze_path_list=vi_haze_val,
                                  train_vi_low_light_path_list=vi_low_light_train,
                                  val_vi_low_light_path_list=vi_low_light_val,
                                  train_vi_rain_path_list=vi_rain_train,
                                  val_vi_rain_path_list=vi_rain_val,
                                  train_vi_haze_low_path_list=train_vi_haze_low_path_list,
                                  val_vi_haze_low_path_list=val_vi_haze_low_path_list,
                                  train_vi_noise_low_path_list=train_vi_noise_low_path_list,
                                  val_vi_noise_low_path_list=val_vi_noise_low_path_list,
                                  train_vi_rain_haze_path_list=train_vi_rain_haze_path_list,
                                  val_vi_rain_haze_path_list=val_vi_rain_haze_path_list,
                                  train_llsn_path_list=train_llsn_path_list,
                                  val_llsn_path_list=val_llsn_path_list,
                                  train_oelc_path_list=train_oelc_path_list,
                                  val_oelc_path_list=val_oelc_path_list,
                                  train_rhrn_path_list=train_rhrn_path_list,
                                  val_rhrn_path_list=val_rhrn_path_list,
                                  phase="train",
                              transform=data_transform["train"])

    val_dataset = PromptDataSet(train_vi_noise_path_list=vi_noise_train,
                                  val_vi_noise_path_list=vi_noise_val,
                                  train_over_exposure_path_list=over_exposure_train,
                                  val_over_exposure_path_list=over_exposure_val,
                                  train_ir_low_contrast_path_list=ir_low_contrast_train,
                                  val_ir_low_contrast_path_list=ir_low_contrast_val,
                                  train_ir_noise_path_list=ir_noise_train,
                                  val_ir_noise_path_list=ir_noise_val,
                                  train_ir_stripe_noise_path_list = ir_stripe_noise_train,
                                  val_ir_stripe_noise_path_list = ir_stripe_noise_val,
                                  train_vi_blur_path_list=vi_blur_train,
                                  val_vi_blur_path_list=vi_blur_val,
                                  train_vi_haze_path_list=vi_haze_train,
                                  val_vi_haze_path_list=vi_haze_val,
                                  train_vi_low_light_path_list=vi_low_light_train,
                                  val_vi_low_light_path_list=vi_low_light_val,
                                  train_vi_rain_path_list=vi_rain_train,
                                  val_vi_rain_path_list=vi_rain_val,
                                  train_vi_haze_low_path_list=train_vi_haze_low_path_list,
                                  val_vi_haze_low_path_list=val_vi_haze_low_path_list,
                                  train_vi_noise_low_path_list=train_vi_noise_low_path_list,
                                  val_vi_noise_low_path_list=val_vi_noise_low_path_list,
                                  train_vi_rain_haze_path_list=train_vi_rain_haze_path_list,
                                  val_vi_rain_haze_path_list=val_vi_rain_haze_path_list,
                                  train_llsn_path_list=train_llsn_path_list,
                                  val_llsn_path_list=val_llsn_path_list,
                                  train_oelc_path_list=train_oelc_path_list,
                                  val_oelc_path_list=val_oelc_path_list,
                                  train_rhrn_path_list=train_rhrn_path_list,
                                  val_rhrn_path_list=val_rhrn_path_list,
                                  phase="val",
                              transform=data_transform["val"])
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model_clip, _ = clip.load("ViT-L/14@336px", device=device)
    model = create_model(model_clip).to(device)

    for param in model.model_clip.parameters():
        param.requires_grad = False

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_ssim_loss, train_max_loss, train_color_loss, train_text_loss, lr = train_one_epoch(model=model,
                                              model_clip=model_clip,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_ssim_loss", train_ssim_loss, epoch)
        tb_writer.add_scalar("train_max_loss", train_max_loss, epoch)
        tb_writer.add_scalar("train_color_loss", train_color_loss, epoch)
        tb_writer.add_scalar("train_text_loss", train_text_loss, epoch)

        if epoch % args.val_every_epcho == 0 and epoch != 0:
            val_loss, val_ssim_loss, val_max_loss, val_color_loss, val_text_loss = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch, lr=lr, filefold_path=file_img_path)

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim_loss, epoch)
            tb_writer.add_scalar("val_max_loss", val_max_loss, epoch)
            tb_writer.add_scalar("val_color_loss", val_color_loss, epoch)
            tb_writer.add_scalar("val_text_loss", val_text_loss, epoch)


            if val_loss < best_val_loss:
                if args.use_dp == True:
                    save_file = {"model": model.module.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                else:
                    save_file = {"model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
                best_val_loss = val_loss
            
            if args.use_dp == True:
                    save_file = {"model": model.module.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
            else:
                    save_file = {"model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=120)

    # set the appropriate batch-size value for your device
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--vi_noise_slight_path', type=str, default="./dataset/WYD_TextDataset/VI_Noise/VI_Noise_slight")
    parser.add_argument('--vi_noise_moderate_path', type=str, default="./dataset/WYD_TextDataset/VI_Noise/VI_Noise_moderate")
    parser.add_argument('--vi_noise_average_path', type=str, default="./dataset/WYD_TextDataset/VI_Noise/VI_Noise_average")
    parser.add_argument('--vi_noise_serious_path', type=str, default="./dataset/WYD_TextDataset/VI_Noise/VI_Noise_extreme")

    parser.add_argument('--over_exposure_slight_path', type=str, default="./dataset/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_slight")
    parser.add_argument('--over_exposure_moderate_path', type=str, default="./dataset/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_moderate")
    parser.add_argument('--over_exposure_average_path', type=str, default="./dataset/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_average")
    parser.add_argument('--over_exposure_serious_path', type=str, default="./dataset/WYD_TextDataset/VI_Over_exposure/VI_Over_exposure_extreme")

    parser.add_argument('--vi_blur_slight_path', type=str, default="./dataset/WYD_TextDataset/VI_Blur/VI_Blur_slight")
    parser.add_argument('--vi_blur_moderate_path', type=str, default="./dataset/WYD_TextDataset/VI_Blur/VI_Blur_moderate")
    parser.add_argument('--vi_blur_average_path', type=str, default="./dataset/WYD_TextDataset/VI_Blur/VI_Blur_average")
    parser.add_argument('--vi_blur_serious_path', type=str, default="./dataset/WYD_TextDataset/VI_Blur/VI_Blur_extreme")

    parser.add_argument('--vi_haze_slight_path', type=str, default="./dataset/WYD_TextDataset/VI_Haze/VI_Haze_slight")
    parser.add_argument('--vi_haze_moderate_path', type=str, default="./dataset/WYD_TextDataset/VI_Haze/VI_Haze_moderate")
    parser.add_argument('--vi_haze_average_path', type=str, default="./dataset/WYD_TextDataset/VI_Haze/VI_Haze_average")
    parser.add_argument('--vi_haze_serious_path', type=str, default="./dataset/WYD_TextDataset/VI_Haze/VI_Haze_extreme")

    parser.add_argument('--vi_rain_slight_path', type=str, default="./dataset/WYD_TextDataset/VI_Rain/VI_Rain_slight")
    parser.add_argument('--vi_rain_moderate_path', type=str, default="./dataset/WYD_TextDataset/VI_Rain/VI_Rain_moderate")
    parser.add_argument('--vi_rain_average_path', type=str, default="./dataset/WYD_TextDataset/VI_Rain/VI_Rain_average")
    parser.add_argument('--vi_rain_serious_path', type=str, default="./dataset/WYD_TextDataset/VI_Rain/VI_Rain_extreme")

    parser.add_argument('--vi_low_light_slight_path', type=str, default="./dataset/WYD_TextDataset/VI_Low_light/VI_Low_light_slight")
    parser.add_argument('--vi_low_light_moderate_path', type=str, default="./dataset/WYD_TextDataset/VI_Low_light/VI_Low_light_moderate")
    parser.add_argument('--vi_low_light_average_path', type=str, default="./dataset/WYD_TextDataset/VI_Low_light/VI_Low_light_average")
    parser.add_argument('--vi_low_light_serious_path', type=str, default="./dataset/WYD_TextDataset/VI_Low_light/VI_Low_light_extreme")

    parser.add_argument('--vi_haze_low_path', type=str, default="./dataset/WYD_TextDataset/VI_Haze_Low")
    parser.add_argument('--vi_noise_low_path', type=str, default="./dataset/WYD_TextDataset/VI_Noise_Low")
    parser.add_argument('--vi_rain_haze_path', type=str, default="./dataset/WYD_TextDataset/VI_Rain_Haze")
    parser.add_argument('--llsn', type=str, default="./dataset/WYD_TextDataset/llsn")
    parser.add_argument('--oelc', type=str, default="./dataset/WYD_TextDataset/oelc")
    parser.add_argument('--rhrn', type=str, default="./dataset/WYD_TextDataset/rhrn")


    parser.add_argument('--ir_low_contrast_slight_path', type=str, default="./dataset/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_slight")
    parser.add_argument('--ir_low_contrast_moderate_path', type=str, default="./dataset/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_moderate")
    parser.add_argument('--ir_low_contrast_average_path', type=str, default="./dataset/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_average")
    parser.add_argument('--ir_low_contrast_serious_path', type=str, default="./dataset/WYD_TextDataset/IR_Low_contrast/IR_Low_contrast_extreme")


    parser.add_argument('--ir_noise_slight_path', type=str, default="./dataset/WYD_TextDataset/IR_Noise/IR_Noise_slight")
    parser.add_argument('--ir_noise_moderate_path', type=str, default="./dataset/WYD_TextDataset/IR_Noise/IR_Noise_moderate")
    parser.add_argument('--ir_noise_average_path', type=str, default="./dataset/WYD_TextDataset/IR_Noise/IR_Noise_average")
    parser.add_argument('--ir_noise_serious_path', type=str, default="./dataset/WYD_TextDataset/IR_Noise/IR_Noise_extreme")

    parser.add_argument('--ir_stripe_noise_slight_path', type=str, default="./dataset/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_slight")
    parser.add_argument('--ir_stripe_noise_moderate_path', type=str, default="./dataset/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_moderate")
    parser.add_argument('--ir_stripe_noise_average_path', type=str, default="./dataset/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_average")
    parser.add_argument('--ir_stripe_noise_serious_path', type=str, default="./dataset/WYD_TextDataset/IR_Stripe_noise/IR_Stripe_noise_extreme")


    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--val_every_epcho', type=int, default=2, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default = True, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0,1,2,3', help='device id (i.e. 0, 1, 2 or 3)')

    opt = parser.parse_args()

    main(opt)
