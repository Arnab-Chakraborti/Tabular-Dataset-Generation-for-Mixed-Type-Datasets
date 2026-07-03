import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTokenizer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token):
        super().__init__()
        self.num_numeric= num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.num_layers = nn.ModuleList([nn.Linear(1,d_token) for _ in range(num_numeric)])
        self.cat_layers= nn.ModuleList([nn.Linear(cardinality, d_token) for cardinality in cat_cardinalities])

    def forward(self, x_num, x_cat_ohe_list):
        tokens=[]
        for i in range(self.num_numeric):
            col_data= x_num[:,i:i+1]
            tokens.append(self.num_layers[i](col_data).unsqueeze(1))
        for i, x_cat_ohe in enumerate(x_cat_ohe_list):
            tokens.append(self.cat_layers[i](x_cat_ohe).unsqueeze(1))

        return torch.cat(tokens,dim=1)

class TransformersBlock(nn.Module):
    def __init__(self, d_model=4, d_ffn=128):
        super().__init__()
        self.attn = nn. MultiheadAttention(embed_dim= d_model, num_heads=1, batch_first=True)
        self.norm1= nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
            )
        self.norm2=nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _= self.attn(x,x,x)
        x=self.norm1(x+attn_out)
        ffn_out=self.ffn(x)
        x=self.norm2(x+ffn_out)
        return x


class TabSynVAE(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token=4, d_ffn= 128):
        super().__init__()
        self.num_numeric= num_numeric
        self.cat_cardinalities= cat_cardinalities
        self.num_cols=num_numeric+len(cat_cardinalities)
        self.d_token= d_token
        self.tokenizer = FeatureTokenizer(num_numeric, cat_cardinalities, d_token)
        self.encoder_mu= nn.Sequential(
            TransformersBlock(d_token, d_ffn),
            TransformersBlock(d_token, d_ffn)
            )
        self.encoder_logvar = nn.Sequential(
            TransformersBlock(d_token, d_ffn),
            TransformersBlock(d_token, d_ffn)
            )
        self.decoder = nn.Sequential(
            TransformersBlock(d_token, d_ffn),
            TransformersBlock(d_token, d_ffn)
            )
        self.detok_num= nn.ModuleList([nn.Linear(d_token,1) for _ in range(num_numeric)])
        self.detok_cat=nn.ModuleList([nn.Linear(d_token, card) for card in cat_cardinalities])


    def encode(self,x_num, x_cat,x_cat_ohe_list):
        E=self.tokenizer(x_num, x_cat_ohe_list)
        mu= self.encoder_mu(E)
        logvar= self.encoder_logvar(E)
        return mu, logvar

    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std

    def decode(self,Z_matrix):
        E_hat= self.decoder(Z_matrix)
        x_num_hat= []
        x_cat_logits_hat = []

        for i in range(self.num_numeric):
            x_num_hat.append(self.detok_num[i](E_hat[:,i]))

        for i in range(len(self.cat_cardinalities)):
            col_idx=self.num_numeric+i
            x_cat_logits_hat.append(self.detok_cat[i](E_hat[:, col_idx]))

        return torch.cat(x_num_hat,dim=-1) if x_num_hat else None, x_cat_logits_hat

    def forward(self,x_num, x_cat_ohe_list):
        mu, logvar=self.encode(x_num, x_cat_ohe_list)
        z= self.reparameterize(mu,logvar)
        x_num_hat,x_cat_logits_hat=self.decode(z)
        return x_num_hat, x_cat_logits_hat, mu, logvar
            
        
    
