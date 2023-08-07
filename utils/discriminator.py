
import torch.nn as nn

class Discriminator(nn.Module):
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
        flatten_x = conv3.reshape(conv3.shape[0], conv3.shape[1])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.linear3(out_2)
        out = self.sigmoid(out_3)
        return out