#Mobilenet_v2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super(MobileNetFeatureExtractor, self).__init__()
        self.device = device
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        with torch.no_grad():
            x = self.transform(image).unsqueeze(0).to(self.device)
            features = self.feature_extractor(x)
            pooled = self.pool(features)
            return pooled.view(-1)












#Optimizasyondan Önce
#Mobilenet_v2
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = mobilenet_v2(pretrained=True)#Bu model değiştirilcek
        self.model.classifier = torch.nn.Identity()  # sınıflandırıcı katmanı çıkar
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(img)
        return embedding.squeeze().cpu().numpy()
"""