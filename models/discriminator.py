import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, text_dim, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * 64 * 64 + text_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        x = torch.cat((img.view(img.size(0), -1), text_embedding), dim=1)
        return self.model(x)
