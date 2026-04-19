import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
# 假设 dataset 和 model 在其他文件中，保持原本的导入不变
from dataset import DegradationDataset
from model import Classification

def train(model, dataloader, optimizer, device, model_clip, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_cos_loss = 0.0
    running_mse_loss = 0.0

    with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (img1_tensor, img2_tensor, text_tensor) in enumerate(pbar):
            img1_tensor = img1_tensor.to(device)
            img2_tensor = img2_tensor.to(device)
            text = clip.tokenize(text_tensor).to(device)

            with torch.no_grad():
                text_encoded = model_clip.encode_text(text).to(device)

            optimizer.zero_grad()

            fre = model(img1_tensor, img2_tensor)
            fre = fre.float()
            text_encode = text_encoded.float()
            cos_sim = F.cosine_similarity(fre, text_encode, dim=-1)
            loss_cosine = 1 - cos_sim.mean()
            loss_mse = F.mse_loss(fre, text_encode)
            loss = 3.0 * loss_cosine + 1.0 * loss_mse
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cos_loss += loss_cosine.item()
            running_mse_loss += loss_mse.item() # 记录 MSE
            pbar.set_postfix(
                loss=loss.item(), 
                avg_cos=running_cos_loss / (batch_idx + 1),
                avg_mse=running_mse_loss / (batch_idx + 1)
            )

    avg_loss = running_loss / len(dataloader)
    print(f"\nEpoch {epoch + 1} - Total Loss: {avg_loss:.4f}")

    return avg_loss

def main():
    # 配置
    img_dir = ""
    batch_size = 8
    num_epochs = 100
    target_size = (224, 224)
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    dataset = DegradationDataset(img_dir, target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model_clip, _ = clip.load("ViT-L/14@336px", device=device)
    model = Classification().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = train(model, dataloader, optimizer, device, model_clip, epoch, num_epochs)

        if (epoch + 1) % 5 == 0: 
            torch.save(model.state_dict(), f"CAmodel_epoch_{epoch+1}.pth")
            print(f"Model saved after epoch {epoch+1}")

if __name__ == "__main__":
    main()
