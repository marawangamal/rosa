import torch
import torch.nn as nn
class Model(torch.nn):
    def __int__(self):
        self.deconvolution_sequence = nn.Sequential(
            torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=4),  # [B, W, H, D]
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=4),  # [B, W*2, H*2, D]
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=4),  # [B, W*4, H*4, D]
        )

    def forward(self, x):
        x = self.byteformer(x)  # [B, D]
        batch_size, embedding_dimension = x.shape
        x = x.reshape(batch_size, 1, 1, embedding_dimension)  # [B, 1, 1, 1]
        x = self.deconvolution_sequence(x)  # [B, W, H, D]


# # Sequence embedding
# x = self.conv1d(x)  # [B, T//2, D]
# x = self.conv1d(x)  # [B, T//4, D]
# x = torch.nn.functional.avg_pool1d(x)  # [B, D]