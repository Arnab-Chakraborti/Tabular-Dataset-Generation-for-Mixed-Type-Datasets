import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularEncoder(nn.module):
    def __init__(self, input_dim: int, hidden_dim: int, laten_dim: int):
        super().__init__()
        layers= []
        in_dim= input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim,h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.leakyReLU(0.2))
            in_dim=h_dim
        self.encoder_base= nn.Sequential(*layers) #(*layers) implies unpacking [layers] as layer1-->layer2-->....-->layer n
        #bifurcation into mu and sigma
        self.fc_mu= nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar= nn.Linear(hidden_dims[-1],latent_dim)
        
    def forward(self. x: torch.Tensor):
        hidden = self.encoder_base(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class TabularDecoder(nn.module):

    def __init__(self, latent_dim: int, hidden_dims: list, continuous_dim: int, cardinalities: list, tau: float =1.0):
        super().__init__()
        self.continuous_dim= continuous_dim
        self.tau= tau
        layers=[]
        hidden_dims= hidden_dims[::-1]
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim)
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim=h_dim
        self.decoder_base(*layers)

        if (self.continuous_dim>0):
            self.continuous_head= nn.Linear(hidden_dims[-1], continuous_dim)

        self.categorical_head=nn.ModuleList([nn.Linear(hidden_dims[-1], card) for card in cardinalities])

    def forward(self, z: torch.Tensor)-> torch.Tensor :
        hidden=self.decoder_base(z)
        outputs=[]

        if self.continuous_dim >0:
            outputs.append(self.continuous_head(hidden))
        for head in self.categorical_head:
            logits=head(hidden)
            cat_output=F.gumbel_softmax(logits, tau= self.tau, hard=True)
            outputs= torch.cat(outputs, dim=1)

class MixedTabularVAE(nn.module):#master architecture tying the probabilistic pipeline together

    def __init__(self, input_dim: int, continuous_dim: int, cardinalities: int, hidden_dims= list =[256, 128, 64], latent_dim: int= 32, tau: float=1.0):
        self.encoder= TabularEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder= TabularDecoder(latent_dim, hidden_dims, continuous_dim, cardinalities, tau)

    def reparameterize(self, mu: torch.Tensor, logvar= torch.Tensor) -> torch.Tensor:
        std=torch.exp(0.5 * logvar)
        eps=torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu,logvar= self.encoder(x)
        z= self.reparameterize(mu, logvar)
        recon_x=self.decoder(z)
        return recon_x, z
ß

@torch.no_grad()
def generate(self, num_samples: int, device: str='cpu') -> torch.Tensor:
    self.eval()
    z_prior= torch.randn(num_samples, self.encoder.fc.mu.out_features).to(device)
    synthetic_data = self.decoder(z_prior)
    self.train()

    return synthetic_data
            
