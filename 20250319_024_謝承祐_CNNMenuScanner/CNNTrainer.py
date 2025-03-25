import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

print("Current working directory:", os.getcwd())

# Define CNN Model
class TallyMarkCNN(nn.Module):
    def __init__(self):
        super(TallyMarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input channel is 1 (grayscale image)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)  # Adjusted to 25x25 after two pooling layers
        self.fc2 = nn.Linear(128, 6)  # output classes is 6 (0 to 5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        
        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                if os.path.isfile(img_path):
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Skip invalid images
                        self.images.append((img_path, self.class_to_idx[cls]))
                    except Exception as e:
                        print(f"跳過無效圖片: {img_path}, 錯誤: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((25, 25)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

base_dir = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_dir, 'data', 'train_data')

if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"訓練資料路徑 {train_data_path} 不存在！請確認資料夾結構。")

train_dataset = CustomImageFolder(root=train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = TallyMarkCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the model
model_save_path = os.path.join(base_dir, 'models', 'tally_mark_cnn.pt')
torch.save(model.state_dict(), model_save_path)
print(f"模型已儲存為 {model_save_path}")