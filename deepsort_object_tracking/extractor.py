import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2


class ResNetBlock(nn.Module):
    """Small residual block for Re-ID CNN."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))   # skip connection!


class ReIDNet(nn.Module):
    """
    Lightweight Re-ID CNN.
    Input:  (B, 3, 128, 64) — person crop
    Output: (B, 128)        — appearance embedding
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # 64×32

            # Block 2
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResNetBlock(64),
            nn.MaxPool2d(2, 2),                # 32×16

            # Block 3
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResNetBlock(128),
            nn.MaxPool2d(2, 2),                # 16×8

            # Block 4
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResNetBlock(128),
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.embed   = nn.Linear(128, 128)

    def forward(self, x):
        x = self.backbone(x)       # (B, 128, 16, 8)
        x = self.pool(x)           # (B, 128, 1, 1)
        x = x.flatten(1)           # (B, 128)
        x = self.embed(x)          # (B, 128)
        # L2 normalize → unit vectors for cosine distance
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class FeatureExtractor:
    """
    Extracts appearance embeddings from detection crops.
    """

    def __init__(self, model_path=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model = ReIDNet().to(device)
        self.model.eval()

        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=device)
            )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),   # height×width for person
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def __call__(self, frame, bboxes):
        """
        Extract features for all detections in one frame.

        frame:  BGR image (H, W, 3)
        bboxes: list of [x1, y1, x2, y2]

        Returns: (N, 128) numpy array
        """
        if len(bboxes) == 0:
            return np.array([])

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]   # BGR crop

            if crop.size == 0:
                crop = np.zeros((128, 64, 3), dtype=np.uint8)

            # BGR → RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(self.transform(crop))

        # Stack into batch
        batch = torch.stack(crops).to(self.device)   # (N, 3, 128, 64)

        # Extract embeddings
        embeddings = self.model(batch)                # (N, 128)
        return embeddings.cpu().numpy()
