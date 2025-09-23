import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, output_dim=768):
        super(Adapter, self).__init__()
        self.global_cnn = SmallCNN()
        self.local_cnn = LocalCNN()
        self.freq_cnn = LargeCNN()


        self.fusion_module = BranchAttentionFusion(num_branches=3, feature_dim=1024)


        self.fc1 = nn.Linear(3072, 1536)
        self.fc2 = nn.Linear(1536, 768)
        self.gelu = nn.GELU()

    def forward(self, img):
        if img.shape[1] == 3:
            img = img.mean(dim=1, keepdim=True)

        global_feat = self.global_cnn(img)  # [B, 1024]
        local_feat = self.local_cnn(img)  # [B, 1024]
        freq_feat = self.freq_cnn(get_frequency_domain_tensor(img))  # [B, 1024]


        fused = self.fusion_module(global_feat, local_feat, freq_feat)  # [B, 3072]

        output = self.fc1(fused)
        output = self.gelu(output)
        output = self.fc2(output)
        return output


class BranchAttentionFusion(nn.Module):
    """

    """

    def __init__(self, num_branches=3, feature_dim=1024, reduction=2):
        """

        """
        super(BranchAttentionFusion, self).__init__()
        self.num_branches = num_branches
        self.feature_dim = feature_dim

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_branches, num_branches // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_branches // reduction, num_branches, bias=False),
            nn.Sigmoid()
        )

    def forward(self, *features):
        """

        """
        if len(features) != self.num_branches:
            raise ValueError(f"except {self.num_branches} but get {len(features)} 个。")

        stacked_features = torch.stack(features, dim=1)
        b, c, l = stacked_features.shape  # b: batch_size, c: num_branches, l: feature_dim

        y = self.avg_pool(stacked_features).view(b, c)

        branch_weights = self.fc(y).view(b, c, 1)  # -> (B, 3, 1)
        weighted_features = stacked_features * branch_weights.expand_as(stacked_features)

        fused = weighted_features.view(b, c * l)  # (B, 3 * 1024) -> (B, 3072)
        return fused






class SmallCNN(nn.Module):
    """

    """

    def __init__(self, output_dim=1024):
        super(SmallCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LocalCNN(nn.Module):
    """

    """

    def __init__(self, output_dim=1024):
        super(LocalCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256 * 4, output_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        h_half, w_half = h // 2, w // 2
        patches = [
            x[:, :, 0:h_half, 0:w_half],
            x[:, :, 0:h_half, w_half:w],
            x[:, :, h_half:h, 0:w_half],
            x[:, :, h_half:h, w_half:w]
        ]
        features = [self.conv(patch) for patch in patches]
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class LargeCNN(nn.Module):
    """

    """

    def __init__(self, output_dim=1024):
        super(LargeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_frequency_domain_tensor(image):
    """

    """
    if image is None:
        raise ValueError("None")
    if len(image.shape) != 4 or image.shape[1] != 1:
        raise ValueError(f"should be [b, 1, h, w], but get {image.shape}")

    f = torch.fft.fft2(image, dim=(-2, -1))
    fshift = torch.fft.fftshift(f, dim=(-2, -1))
    magnitude_spectrum = torch.abs(fshift)
    magnitude_spectrum = torch.log(magnitude_spectrum + 1)
    return magnitude_spectrum


