import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, output_dim=768):
        super(Adapter, self).__init__()
        self.global_cnn = SmallCNN()
        self.local_cnn = LocalCNN()
        self.freq_cnn = LargeCNN()

        self.fc1 = nn.Linear(3072, 1536)
        self.fc2 = nn.Linear(1536, 768)
        self.gelu = nn.GELU()

    def forward(self, img1, img2):
        # 1. Preprocessing: RGB to Grayscale
        if img1.shape[1] == 3:
            img1 = img1.mean(dim=1, keepdim=True)  # Shape: [B, 1, H, W]
        if img2.shape[1] == 3:
            img2 = img2.mean(dim=1, keepdim=True)  # Shape: [B, 1, H, W]

        # 2. Spatial Feature Extraction
        # Concatenate along the channel dimension for spatial processing
        img_spatial = torch.cat((img1, img2), dim=1) # Shape: [B, 2, H, W]
        global_feat = self.global_cnn(img_spatial)
        local_feat = self.local_cnn(img_spatial)

        # 3. Frequency Feature Extraction (MODIFIED according to Option 1A)
        # First, get the frequency domain for each image separately
        freq1 = get_frequency_domain_tensor(img1)
        freq2 = get_frequency_domain_tensor(img2)
        # Then, concatenate the frequency tensors along the channel dimension
        freq_combined = torch.cat((freq1, freq2), dim=1) # Shape: [B, 2, H, W]
        # Pass the 2-channel frequency tensor to the CNN
        freq_feat = self.freq_cnn(freq_combined)

        # 4. Fusion and Final Output
        fused = torch.cat([global_feat, local_feat, freq_feat], dim=1)  # [B, 3072]

        output = self.fc1(fused)
        output = self.gelu(output)
        output = self.fc2(output)
        return output

# --- Sub-Modules ---

class SmallCNN(nn.Module):
    """
    """
    def __init__(self, output_dim=1024):
        super(SmallCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # MODIFIED: Input channel is now 2 to handle the concatenated spatial images
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),  # [b, 64, h/2, w/2]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [b, 128, h/4, w/4]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [b, 256, h/8, w/8]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [b, 256, 1, 1]
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
            # MODIFIED: Input channel is now 2, as it processes patches from the concatenated image
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256 * 4, output_dim)  # 4个子图拼接

    def forward(self, x):
        # x is expected to be [B, 2, H, W] here
        b, c, h, w = x.shape
        h_half, w_half = h // 2, w // 2
        patches = [
            x[:, :, 0:h_half, 0:w_half],
            x[:, :, 0:h_half, w_half:w],
            x[:, :, h_half:h, 0:w_half],
            x[:, :, h_half:h, w_half:w]
        ]
        # Each patch is [B, 2, H/2, W/2]
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
            # MODIFIED: Input channel is now 2 to handle the concatenated frequency spectra
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),
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

# --- Helper Function (No changes needed here) ---

def get_frequency_domain_tensor(image):
    """

    """
    if image is None:
        raise ValueError("input image should be None")
    if len(image.shape) != 4 or image.shape[1] != 1:
        raise ValueError(f"should be [b, 1, h, w], but get {image.shape}")

    f = torch.fft.fft2(image, dim=(-2, -1))
    fshift = torch.fft.fftshift(f, dim=(-2, -1))
    magnitude_spectrum = torch.abs(fshift)
    # Add a small epsilon to avoid log(0)
    magnitude_spectrum = torch.log(magnitude_spectrum + 1e-8)
    return magnitude_spectrum