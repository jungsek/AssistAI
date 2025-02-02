import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
import mediapipe as mp


class imageProcessor():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

    def __detect_hand_mediapipe(self, img):
        """Detect and crop hand using MediaPipe Hands."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])

                h, w, _ = img.shape
                x_min, y_min = int(x_min * w), int(y_min * h)
                x_max, y_max = int(x_max * w), int(y_max * h)

                # Crop hand region with padding
                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                cropped_img = img[y_min:y_max, x_min:x_max]
                cropped_img = cv2.resize(cropped_img, (128, 128))
                return cropped_img

        return img  # Return original image if no hand is detected

    def processImage(self, img):
        """Preprocess images: Detect & crop hand, apply edge detection"""
        cropped_hand = self.__detect_hand_mediapipe(img)
        gray_img = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)
        return self.transform(edges).unsqueeze(0)

class cnnModel():
    def __init__(self, C=1, D=64*64, classes=29, filters=64):
        self.model = nn.Sequential(
            nn.Conv2d(C, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(filters, 2*filters, 3, padding=1),
            nn.BatchNorm2d(2*filters),
            nn.ReLU(),
            nn.Conv2d(2*filters, 2*filters, 3, padding=1),
            nn.BatchNorm2d(2*filters),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(2*filters, 4*filters, 3, padding=1),
            nn.BatchNorm2d(4*filters),
            nn.ReLU(),
            nn.Conv2d(4*filters, 4*filters, 3, padding=1),
            nn.BatchNorm2d(4*filters),
            nn.ReLU(),
            nn.Conv2d(4*filters, 4*filters, 3, padding=1),
            nn.BatchNorm2d(4*filters),
            nn.ReLU(),
            nn.Conv2d(4*filters, 4*filters, 3, padding=1),
            nn.BatchNorm2d(4*filters),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(4*filters, 8*filters, 3, padding=1),
            nn.BatchNorm2d(8*filters),
            nn.ReLU(),
            nn.Conv2d(8*filters, 8*filters, 3, padding=1),
            nn.BatchNorm2d(8*filters),
            nn.ReLU(),
            nn.Conv2d(8*filters, 8*filters, 3, padding=1),
            nn.BatchNorm2d(8*filters),
            nn.ReLU(),
            nn.Conv2d(8*filters, 8*filters, 3, padding=1),
            nn.BatchNorm2d(8*filters),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),
            nn.Linear(8 * filters * D // (16**2), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, classes)
        )

        self.classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

    def loadWeights(self, path):
        """Load trained weights from a specified file"""
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    def predictImage(self, img):
        """Return the class probabilities for a 64x64, 1 channel (black and white), edge detected image"""
        #set the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            output = self.model(img)
            #apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1) 
            #convert to a list of (class_index, probability) tuples
            return [(self.classNames[i], round(prob.item(),5)) for i, prob in enumerate(probabilities.squeeze())]
