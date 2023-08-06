import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.extend((
                nn.Linear(config[i - 1], config[i]),
                nn.ReLU()
            ))
        
        self.config = config
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], config[-1])
        self.fc_var = nn.Linear(config[-1], config[-1])

        modules = []
        for i in range(len(config) - 1, 0, -1):
            modules.extend((
                nn.Linear(config[i], config[i - 1]),
                nn.ReLU()
            ))
        modules[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5* logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar
    
if __name__ == "__main__":
    config = [10, 784, 512, 256, 128, 64, 32, 16, 8]
    model = VAE(config)
    print(model)