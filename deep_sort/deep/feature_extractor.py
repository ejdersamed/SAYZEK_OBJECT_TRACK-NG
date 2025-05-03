#RESNET
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super(ResNet18FeatureExtractor, self).__init__()
        self.device = device
        # Önceden eğitilmiş ResNet-18 modelini yükle
        resnet18 = models.resnet18(pretrained=True)
        # Son sınıflandırma katmanını (fc) kaldır
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Görüntü ön işleme adımları
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet-18 için uygun boyut
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
       
        with torch.no_grad():
            # Görüntüyü ön işle
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            # Özellikleri çıkar
            features = self.feature_extractor(img_tensor)
            # [1, 512, 1, 1] boyutundaki çıktıyı [512] boyutuna getir
            features = features.view(features.size(0), -1)
            return features.squeeze(0)
