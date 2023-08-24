
import torch
import torch.nn as nn
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        modules = []
        modules.append(
            SelfAttention(config[0])
        )
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
    
    def loss_function(self, recon_x, x, mu, logVar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        return BCE + KLD
    

class SelfAttention(pl.LightningModule):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, V)
        return attended_values


class Discriminator(pl.LightningModule):
    def __init__(self, input_size, kernel_size = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size = kernel_size, stride = 1, padding = 'same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size = kernel_size, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = kernel_size, stride = 1, padding = 'same')
        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x = conv3.squeeze()
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.linear3(out_2)
        out = self.sigmoid(out_3)
        return out
    

class Generator(pl.LightningModule):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, noise=None):
        h0 = torch.zeros((1, x.size(0), 1024), device=self.device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros((1, x.size(0), 512), device=self.device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros((1, x.size(0), 256), device=self.device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out