# model.py
import torch
import torch.nn as nn

class DigitGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_size=28):
        super(DigitGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, img_size * img_size),
            nn.Tanh()
        )
        self.img_size = img_size

    def forward(self, z, labels):
        labels = self.label_emb(labels)
        x = torch.cat([z, labels], dim=1)
        img = self.model(x)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img
