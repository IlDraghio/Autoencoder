from torch import nn

class AE(nn.Module):
    def __init__(self, hidden_channels=16,kernel_size=4,stride_size=3,latent_dim=256):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(

          nn.Conv1d(1, hidden_channels, kernel_size, stride_size,padding=4,dilation=2),
          nn.ReLU(),
          nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size, stride_size,padding=3,dilation=2),
          nn.ReLU(),
          nn.Conv1d(hidden_channels*2, hidden_channels*4, kernel_size, stride_size,padding=0,dilation=1),
          nn.ReLU(),
          nn.Conv1d(hidden_channels*4, hidden_channels*8, kernel_size, stride_size,padding=2,dilation=1),
          nn.ReLU(),
        )

        self.fc1 = nn.Linear(5120, latent_dim)
        self.fc2 = nn.Linear(latent_dim,5120)

        self.decoder = nn.Sequential(
          nn.ConvTranspose1d(hidden_channels*8,hidden_channels*4,kernel_size,stride_size,padding=2,output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose1d(hidden_channels*4,hidden_channels*2,kernel_size,stride_size,padding=0,output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose1d(hidden_channels*2,hidden_channels,kernel_size,stride_size,padding=1,output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose1d(hidden_channels,1,kernel_size,stride_size,padding=3,output_padding=1),
          nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.encoder(x)

        dimensions = x.shape[1]
        n_points = x.shape[2]

        x = x.view(x.shape[0],-1)

        x = self.fc1(x)
        latent_space = x
        x = self.fc2(x)

        x = x.view(x.shape[0] ,dimensions ,n_points)

        x_recon = self.decoder(x)

        return x_recon, latent_space