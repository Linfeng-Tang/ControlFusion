import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from collections import defaultdict


class DegradationDataset(Dataset):
    def __init__(self, img_dir, target_size=(256, 256)):
        self.img_dir = img_dir
        self.target_size = target_size
        self.image_pairs = self._create_pairs()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        ir_path, vi_path = self.image_pairs[idx]

        # 读取图像
        ir_img = Image.open(ir_path).convert('RGB')
        vi_img = Image.open(vi_path).convert('RGB')

        # 从文件名对中生成单一的描述信息
        description = self._get_single_description_for_pair(ir_path, vi_path)

        # 转换为Tensor
        ir_tensor = self.transform(ir_img)
        vi_tensor = self.transform(vi_img)

        return ir_tensor, vi_tensor, description

    def _create_pairs(self):
        image_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]
        image_groups = defaultdict(list)
        for f in image_files:
            try:
                img_id = f.split('_')[0]
                image_groups[img_id].append(f)
            except IndexError:
                print(f"Warning: Skipping file {f} due to unexpected naming format.")
                continue

        valid_pairs = []
        for img_id, files in image_groups.items():
            parsed_files = {}
            for f in files:
                try:
                    parts = f.split('.')[0].split('_')
                    modality, degradation = parts[1], parts[2]
                    parsed_files[(modality, degradation)] = os.path.join(self.img_dir, f)
                except IndexError:
                    continue

            special_cases = [
                (('ir', 'lowcontrast'), ('vi', 'overexposure')),
                (('ir', 'noise'), ('vi', 'rainhaze')),
                (('ir', 'stripenoise'), ('vi', 'lowlight'))
            ]
            for ir_key, vi_key in special_cases:
                if ir_key in parsed_files and vi_key in parsed_files:
                    valid_pairs.append((parsed_files[ir_key], parsed_files[vi_key]))

            degradations = set(key[1] for key in parsed_files.keys() if key[1] != 'normal')
            for deg in degradations:
                if ('vi', deg) in parsed_files and ('ir', 'normal') in parsed_files:
                    valid_pairs.append((parsed_files[('ir', 'normal')], parsed_files[('vi', deg)]))
                if ('ir', deg) in parsed_files and ('vi', 'normal') in parsed_files:
                    valid_pairs.append((parsed_files[('ir', deg)], parsed_files[('vi', 'normal')]))

        return sorted(list(set(valid_pairs)))

    def _get_single_description_for_pair(self, ir_path, vi_path):
        def parse_path(path):
            base_name = os.path.basename(path).split('.')[0]
            parts = base_name.split('_')
            return parts[1], parts[2]  # modality, degradation

        ir_mod, ir_deg = parse_path(ir_path)
        vi_mod, vi_deg = parse_path(vi_path)

        # 规则1: 检查是否为特殊组合
        # 为了保证顺序无关，我们将退化类型按字母排序后组合
        pair_key = tuple(sorted((ir_deg, vi_deg)))

        # 特殊情况1：vi过曝 & ir低对比度
        if pair_key == ('lowcontrast', 'overexposure'):
            return self.get_description('combined', 'overexposure_lowcontrast')

        # 特殊情况2：vi雨雾 & ir噪声
        if pair_key == ('noise', 'rainhaze'):
            return self.get_description('combined', 'rainhaze_noise')

        # 特殊情况3：vi低光 & ir条纹噪声
        if pair_key == ('lowlight', 'stripenoise'):
            return self.get_description('combined', 'lowlight_stripenoise')

        # 规则2: 标准情况（一个正常，一个退化）
        if ir_deg != 'normal':
            return self.get_description(ir_mod, ir_deg)
        elif vi_deg != 'normal':
            return self.get_description(vi_mod, vi_deg)
        else:  # 理论上不会发生，因为我们的配对逻辑保证了至少有一个退化（除非有ir_normal和vi_normal的配对）
            return "A pair of normal infrared and visible images."

    def get_description(self, xx, yy):
        description_dict = {
            'combined': {
                'overexposure_lowcontrast': "This is the infrared visible light fusion task, Visible images have the overexposure degradation. Infrared images have the low contrast degradation. ",
                'rainhaze_noise': "In this project, we're working on the fusion of infrared and visible light images, especially addressing the issue of rain and haze in the visible spectrum. Infrared images have the random noise degradation.",
                'lowlight_stripenoise': "lowlight in the visible spectrum is a critical factor to address in the infrared-visible light image fusion task. Infrared images have the stripe noise degradation."
            },

            'ir': {
                'normal': "This is a clean, non-degraded infrared image, serving as a reference for the fusion task.",
                'lowcontrast': "This is the infrared visible light fusion task. Infrared images have the 10 low contrast degradation.",
                'stripenoise': "The challenge at hand is the fusion of infrared and visible light images, with the added complexity of 10 stripe noise degradation in the infrared images.",
                'noise': "We are dealing with the fusion task involving infrared and visible light images, where the infrared imagery is affected by 10 noise degradation."
            },
            'vi': {
                'normal': "This is a clean, non-degraded visible light image, serving as a reference for the fusion task.",
                'noise': "Our focus is on the fusion of infrared and visible light images, particularly addressing the 10 noise degradation in the visible spectrum.",
                'haze': "In the context of infrared-visible light image fusion, we're working on enhancing the quality of visible images suffering from 10 haze degradation.",
                'rain': "The core of this task revolves around combining infrared and visible light images, addressing the limitations of visible imagery in low-light environments.",
                'blur': "We're working on the fusion of infrared and visible light images, where the visible spectrum experiences 10 blur.",
                'hazelow': "Our aim is to fuse infrared and visible light images, particularly addressing the haze and low light in visible images.",
                'noiselow': "The success of the infrared-visible light image fusion task hinges on our ability to correct noise and low light problems in the visible images.",
                'overexposure': "The quality of the fused infrared-visible light images greatly depends on our ability to handle 10 overexposure issues in the visible spectrum.",
                'lowlight': "This is about fusing infrared and visible light images, where the visible ones are plagued by 10 lowlight.",
                'rainhaze': "This task involves merging infrared and visible light images, with visible images experiencing haze and rain problems."
            }
        }

        if xx in description_dict and yy in description_dict[xx]:
            return description_dict[xx][yy]
        else:
            return f"Unknown degradation for {xx} with {yy}."

    def transform(self, img):
        transform_ops = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])
        return transform_ops(img)


# img_dir = '/media/lfw/Data/WYD_Classification_dataset'
# dataset = DegradationDataset(img_dir)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#

# for ir_images, vi_images, descriptions in dataloader:
#     print("--- Batch ---")
#     print("IR Image Batch Shape:", ir_images.shape)
#     print("VI Image Batch Shape:", vi_images.shape)
#     print("\nCombined Descriptions:", descriptions)
#     print("--- End Batch ---\n")

