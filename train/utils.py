import torch

def generate_noise(batch_size, noise_dim):
    return torch.randn(batch_size, noise_dim)

def get_text_embeddings(batch_size, text_dim):
    return torch.randn(batch_size, text_dim)
