import torch, torch.nn as nn, cv2, numpy as np

class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(True),
            nn.Conv2d(128,128,3,padding=1),nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),nn.ReLU(True),
            nn.Conv2d(256,256,3,padding=1),nn.ReLU(True),
            nn.Conv2d(256,256,3,padding=1),nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1),nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512,512,3,dilation=2,padding=2),nn.ReLU(True),
            nn.Conv2d(512,512,3,dilation=2,padding=2),nn.ReLU(True),
            nn.Conv2d(512,512,3,dilation=2,padding=2),nn.ReLU(True),
            nn.Conv2d(512,256,3,dilation=2,padding=2),nn.ReLU(True),
            nn.Conv2d(256,128,3,dilation=2,padding=2),nn.ReLU(True),
            nn.Conv2d(128,64,3,dilation=2,padding=2),nn.ReLU(True),
        )
        self.output_layer = nn.Conv2d(64,1,1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

class CrowdDensityEstimator:
    def __init__(self, weights="models/csrnet_model_best.pth"):
        import os
        self.model = CSRNet()
        if os.path.exists(weights):
            ckpt = torch.load(weights, map_location="cpu")
            self.model.load_state_dict(ckpt)
            self.model.eval()
            print("CSRNet crowd estimator loaded")
            self.ready = True
        else:
            print(f"Warning: '{weights}' not found. Crowd density estimation will be SKIPPED.")
            self.ready = False

    def estimate(self, frame):
        if not self.ready:
            return 0, None
        img = cv2.resize(frame, (640,480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)/255
        with torch.no_grad():
            density = self.model(t).squeeze().numpy()
            count = int(density.sum())
        return count, density
