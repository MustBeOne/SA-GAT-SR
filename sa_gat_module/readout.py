import torch.nn as nn
import torch
from torch_scatter import scatter_add
import torch_geometric.utils as utils

class ReadOut(nn.Module):
    def __init__(self,d_model,dropout=0,num_layers=1,edge_dim=None, global_fea_dim=8, max_nbr_num=12,btc_size=128):
        super().__init__()
        self.in_channels = d_model
        self.out_channels = 2 * self.in_channels
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.out_channels,
                            hidden_size=self.in_channels,
                            num_layers=self.num_layers,
                            batch_first=False,
                            bias=True,
                            bidirectional=False)
        self.Wq=nn.Linear(d_model,self.in_channels,bias=False)
        self.Wk=nn.Linear(edge_dim,self.in_channels,bias=False)
        self.Wv=nn.Linear(edge_dim,self.in_channels,bias=False)
        self.ff=nn.Linear(self.out_channels,self.out_channels)
        self.h0_fea_emb=nn.Linear(global_fea_dim,self.in_channels*num_layers)
        self.global_fea_emb=nn.Linear(global_fea_dim,self.out_channels)
        self.q_star_ln=nn.LayerNorm(self.out_channels)
        self.h0_ln=nn.LayerNorm(self.in_channels)
        self.drop=nn.Dropout(dropout)
        self.lk_relu=nn.LeakyReLU(0.1)
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.ff.weight)
    def forward(self,x, edge_attr, nbr_fea_idx, rank, degree, batch):
        q, self.hn = self.gru(self.q_star, self.hn)
        dim=x.shape[-1]**-0.5

        local_mask=torch.masked_fill(rank,rank>0,-1e6)
        local_mask=torch.masked_fill(local_mask,local_mask>=0,0) 
        query = self.Wq(x)
        dim=query.shape[-1]**-0.5
        key = self.Wk(edge_attr) 
        value = self.Wv(edge_attr)
        a_ij=(query.unsqueeze(1)*key).sum(-1)*dim
        attn_ij=self.drop(torch.softmax(a_ij+local_mask,dim=-1))
        x=x+(attn_ij.unsqueeze(-1)*value).sum(1)
        
        t=torch.sum(q[0][batch]*x,dim=-1)*dim
        e=self.drop(utils.softmax(t,batch))
        r = scatter_add(e.unsqueeze(-1)*x,batch,dim=0).squeeze(-1)
        self.q_star = self.q_star+self.lk_relu(self.ff(torch.cat([q, r.unsqueeze(0)], dim=-1)))
        self.q_star=self.q_star_ln(self.q_star[0]).unsqueeze(0)
        
    def return_output(self):
        return self.q_star.squeeze(0)
    
    def return_output1(self,x, edge_attr, rank,degree, batch):
        local_mask=torch.masked_fill(rank,rank>0,-1e6)
        local_mask=torch.masked_fill(local_mask,local_mask>=0,0) 
        query = self.Wq(x)
        dim=query.shape[-1]**-0.5
        key = self.Wk(edge_attr) 
        value = self.Wv(edge_attr)
        a_ij=(query.unsqueeze(1)*key).sum(-1)*dim
        attn_ij=self.drop(torch.softmax(a_ij+local_mask,dim=-1))
        x=x+(attn_ij.unsqueeze(-1)*value).sum(1)
        for _ in range(3):
            q, self.hn = self.gru(self.q_star, self.hn)
            dim=x.shape[-1]**-0.5
            t=torch.sum(q[0][batch]*x,dim=-1)*dim
            t=t+self.deg_weight(degree-1).squeeze(-1)
            e=self.drop(utils.softmax(t,batch))
            r = scatter_add(e.unsqueeze(-1)*x,batch,dim=0).squeeze(-1)
            self.q_star = torch.cat([q, r.unsqueeze(0)], dim=-1)
        return self.ff(self.q_star.squeeze(0))
    def clear_attr(self):
        delattr(self,'q_star')
        delattr(self,'hn')
    def return_outsize(self):
        return self.out_channels