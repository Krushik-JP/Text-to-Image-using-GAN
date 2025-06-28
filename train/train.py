import torch
from models.generator import Generator
from models.discriminator import Discriminator
from utils import generate_noise, get_text_embeddings

# Hyperparameters
epochs = 20
batch_size = 64
noise_dim = 100
text_dim = 100
img_channels = 3

# Initialize models
G = Generator(noise_dim, text_dim, img_channels)
D = Discriminator(text_dim, img_channels)

# Optimizers
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)

# Loss
criterion = torch.nn.BCELoss()

# Dummy training loop
for epoch in range(epochs):
    for i in range(10):  # Simulated batches
        noise = generate_noise(batch_size, noise_dim)
        text_embed = get_text_embeddings(batch_size, text_dim)

        # Train Discriminator
        real_images = torch.randn(batch_size, img_channels, 64, 64)
        real_labels = torch.ones(batch_size, 1)
        fake_images = G(noise, text_embed).detach()
        fake_labels = torch.zeros(batch_size, 1)

        d_loss_real = criterion(D(real_images, text_embed), real_labels)
        d_loss_fake = criterion(D(fake_images, text_embed), fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        fake_images = G(noise, text_embed)
        g_loss = criterion(D(fake_images, text_embed), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

