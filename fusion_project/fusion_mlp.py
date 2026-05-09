import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, 1),      nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze()

def load_fusion_model(checkpoint_path, device):
    model = FusionMLP()
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval()

def predict(p_deepfake, p_scam, model, device):
    x = torch.tensor([[p_deepfake, p_scam]], dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(x).item()
