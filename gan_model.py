import torch
import torch.nn as nn
from torchinfo import summary

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, d=128, out_channels=3):
        super().__init__()
        
        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Linear layer to process noise + class embedding
        self.input_layer = nn.Linear(latent_dim + num_classes, d*8*4*4)

        # reshape to (32, 1024, 4, 4)
        self.reshape_size = (-1, d*8, 4, 4)
        
        # Transposed convolution layers
        self.main = nn.Sequential(
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),  # [batch, 512, 8, 8]
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),  # [batch, 256, 16, 16]
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*2, out_channels, 4, 2, 1),  # [batch, 3, 32, 32]
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, noise, labels):
        # Process label embedding
        label_embedding = self.label_embedding(labels)
        # print(f'label embed: {label_embedding.shape}')
        
        # noise is [32, 100] + [32, 10]
        x = torch.cat([noise, label_embedding], dim=1)
        
        x = self.input_layer(x)
        # print(x.shape)
        x = x.view(self.reshape_size)
        # print(x.shape)

        x = self.main(x)
        return x

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, d=64, in_channels=3):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, d, 4, 2, 1),  # [batch, 64, 16, 16]
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # (num_embeddings, embedding_dim)
        self.embed = nn.Embedding(num_classes, d)
        
        # Main convolution layers
        self.main = nn.Sequential(
            nn.Conv2d(d + d, d*2, 4, 2, 1),  # [batch, 128, 8, 8]
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(d*2, d*4, 4, 2, 1),  # [batch, 256, 4, 4]
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(d*4, 1, 4, 1, 0)  # [batch, 1, 1, 1]
        )
        
    def forward(self, x, labels):
        batch_size = x.size(0) # 32, but can be different
        first_conv_out = self.initial(x) # Output shape (32, 64, 16, 16)
        
        # Get class embeddings and reshape to spatial feature maps
        y_embedding = self.embed(labels) # Outputs (32, 64)

        y_embedding = y_embedding.view(batch_size, -1, 1, 1) # Outputs (32, 64, 1, 1)
        y_embedding = y_embedding.expand(batch_size, -1, first_conv_out.size(2), first_conv_out.size(3)) # Shape (32, 64, 16, 16)
        
        # Concatenate features and class embeddings
        first_conv_out = torch.cat([first_conv_out, y_embedding], dim=1)
        
        # Process through main layers
        output = self.main(first_conv_out) # Outputs (32, 1, 1, 1)
        
        # Return logits, not sigmoid (for better numerical stability)
        return output.view(batch_size, 1)
    

if __name__ == "__main__":        
    # Testing Conditional Generator
    gan = ConditionalGenerator(latent_dim=100, num_classes=10)
    
    # Note: Conditional generator takes noise vector and labels as separate inputs
    noise = torch.randn(32, 100)  # Flattened noise, not 4D
    labels = torch.randint(0, 10, (32,))  # Random class labels
    
    # Generate images with the conditional generator
    output = gan(noise, labels)
    print("Generator output shape:", output.shape)  # Should be [32, 3, 32, 32]

    # Testing Conditional Discriminator
    disc = ConditionalDiscriminator(num_classes=10)
    image = torch.randn(32, 3, 32, 32)  # Random fake images
    
    # Discriminate images with conditional discriminator
    output = disc(image, labels)
    print("Discriminator output shape:", output.shape)  # Should be [32, 1, 1, 1]

    # summary(gan, input_data=[noise, labels])
    
    # summary(disc, input_data=[image, labels])