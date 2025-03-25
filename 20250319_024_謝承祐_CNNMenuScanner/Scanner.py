import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os

# Define CNN Model
class TallyMarkCNN(nn.Module):
    def __init__(self):
        super(TallyMarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)  # Adjusted to 25x25 after two pooling layers
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'tally_mark_cnn.pt')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型檔案 {model_path} 不存在！請先執行 CNNTrainer.py 進行訓練。")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TallyMarkCNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((25, 25)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

print("按下 'q' 鍵退出程式")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝影機畫面")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi = thresh[y:y+h, x:x+w]

        try:
            roi_resized = cv2.resize(roi, (25, 25))
            input_tensor = transform(roi_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                tally_count = predicted.item()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Count: {tally_count}', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"處理錯誤: {e}")
            continue

    cv2.imshow('Tally Mark Scanner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()