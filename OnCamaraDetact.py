import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# =========================
# CNN MODEL (same as training)
# =========================
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =========================
# DEVICE (GPU preferred)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD MODEL
# =========================
model = DigitCNN().to(device)
model.load_state_dict(
    torch.load("digit_cnn.pth", map_location=device, weights_only=True)
)
model.eval()

# =========================
# OPENCV SETUP
# =========================
WINDOW_NAME = "Digit Recognizer"
cv2.namedWindow(WINDOW_NAME)

canvas = np.zeros((400, 400), dtype=np.uint8)
drawing = False
predicted_digit = None

def draw_digit(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 10, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.setMouseCallback(WINDOW_NAME, draw_digit)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# MAIN LOOP (SINGLE WINDOW)
# =========================
while True:
    # base frame
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # UI text
    cv2.putText(
        frame,
        "Draw digit | P: Predict | C: Clear | ESC: Exit",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # show prediction if exists
    if predicted_digit is not None:
        cv2.putText(
            frame,
            f"Prediction: {predicted_digit}",
            (80, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

    # single window render
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    # CLEAR
    if key == ord('c'):
        canvas[:] = 0
        predicted_digit = None

    # PREDICT
    elif key == ord('p'):
        img = cv2.resize(canvas, (28, 28))
        img = img.astype(np.float32) / 255.0
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            predicted_digit = torch.argmax(output, 1).item()

        print("Predicted Digit:", predicted_digit)

    # EXIT
    elif key == 27:
        break

cv2.destroyAllWindows()
