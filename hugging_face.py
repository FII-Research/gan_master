import torch
from gan_model import ConditionalGenerator, ConditionalDiscriminator
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder
import os
from dotenv import load_dotenv
import json

load_dotenv()

REPO_NAME = os.getenv("REPO_NAME")
HF_USERNAME = os.getenv("HF_USERNAME")
TOKEN = os.getenv("TOKEN")
GEN_MODEL_PATH = os.getenv("GEN_MODEL_PATH")
DISC_MODEL_PATH = os.getenv("DISC_MODEL_PATH")

# Step 1: Load checkpoints
print("📥 Loading checkpoints...")
gen_checkpoint = torch.load(GEN_MODEL_PATH, map_location="cpu")
disc_checkpoint = torch.load(DISC_MODEL_PATH, map_location="cpu")

# Step 2: Rebuild models (must match training config)
print("Rebuilding models...")
generator = ConditionalGenerator(latent_dim=100, num_classes=10)
generator.load_state_dict(gen_checkpoint)
generator.eval()

discriminator = ConditionalDiscriminator(num_classes=10)
discriminator.load_state_dict(disc_checkpoint)
discriminator.eval()

# Step 3: Save models + classes locally
save_dir = "model_repo"
os.makedirs(save_dir, exist_ok=True)

# Save both models in the same directory
torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))

# Save model config
config = {
    "model_type": "ConditionalGAN",
    "latent_dim": 100,
    "num_classes": 10,
    "dataset": "CIFAR-10",
    "generator_architecture": "ConditionalGenerator",
    "discriminator_architecture": "ConditionalDiscriminator",
    "image_size": 32,
    "channels": 3
}

with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Save model class definitions
with open(os.path.join(save_dir, "gan_model.py"), "w") as f:
    f.write('''# GAN Model Definitions
import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, d=128, out_channels=3):
        super().__init__()
        
        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Linear layer to process noise + class embedding
        self.input_layer = nn.Linear(latent_dim + num_classes, d*8*4*4)
        self.reshape_size = (-1, d*8, 4, 4)
        
        # Transposed convolution layers
        self.main = nn.Sequential(
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        x = self.input_layer(x)
        x = x.view(self.reshape_size)
        x = self.main(x)
        return x

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, d=64, in_channels=3):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.embed = nn.Embedding(num_classes, d)
        
        self.main = nn.Sequential(
            nn.Conv2d(d + d, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(d*4, 1, 4, 1, 0)
        )
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        first_conv_out = self.initial(x)
        
        y_embedding = self.embed(labels)
        y_embedding = y_embedding.view(batch_size, -1, 1, 1)
        y_embedding = y_embedding.expand(batch_size, -1, first_conv_out.size(2), first_conv_out.size(3))
        
        first_conv_out = torch.cat([first_conv_out, y_embedding], dim=1)
        output = self.main(first_conv_out)
        
        return output.view(batch_size, 1)
''')

# Create README
readme_content = """# Conditional GAN for CIFAR-10

This repository contains a trained Conditional Generative Adversarial Network (CGAN) for generating CIFAR-10 images.

## Model Details

- **Architecture**: Conditional GAN
- **Dataset**: CIFAR-10
- **Image Size**: 32x32x3
- **Latent Dimension**: 100
- **Number of Classes**: 10

## Usage

```python
import torch
from gan_model import ConditionalGenerator, ConditionalDiscriminator

# Load the models
generator = ConditionalGenerator(latent_dim=100, num_classes=10)
discriminator = ConditionalDiscriminator(num_classes=10)

generator.load_state_dict(torch.load("generator.pth"))
discriminator.load_state_dict(torch.load("discriminator.pth"))

generator.eval()
discriminator.eval()

# Generate images
noise = torch.randn(1, 100)
label = torch.tensor([0])  # Class 0 (airplane)
fake_image = generator(noise, label)

# Evaluate with discriminator
disc_output = discriminator(fake_image, label)
```

## Classes

0. airplane
1. automobile  
2. bird
3. cat
4. deer
5. dog
6. frog
7. horse
8. ship
9. truck

## Files

- `generator.pth`: Trained generator weights
- `discriminator.pth`: Trained discriminator weights
- `gan_model.py`: Model architecture definitions
- `config.json`: Model configuration
"""

with open(os.path.join(save_dir, "README.md"), "w") as f:
    f.write(readme_content)

# Step 4: Create repo on Hugging Face Hub
print("🚀 Creating repository on Hugging Face Hub...")
create_repo(f"{HF_USERNAME}/{REPO_NAME}", token=TOKEN, exist_ok=True)

# Step 5: Upload files
print("�� Uploading files to Hugging Face Hub...")
upload_folder(
    folder_path=save_dir,
    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
    token=TOKEN,
    path_in_repo="."
)

print(f"✅ Models uploaded successfully!")
print(f"�� View at: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
print(f"�� Files uploaded:")
print(f"   - generator.pth")
print(f"   - discriminator.pth")
print(f"   - config.json")
print(f"   - gan_model.py")
print(f"   - README.md")