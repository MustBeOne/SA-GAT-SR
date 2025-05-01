# -*- coding: utf-8 -*-
from torch import nn
import torch
from torch_scatter import scatter_mean
from .modules import SAGATEncoder, SelfAttentionEncoder
from .readout import ReadOut

class GaussianDistance(object):
    def __init__(self,dmin,dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax+step, step) if (dmax-dmin)/step<32 else torch.linspace(dmin, dmax, 32)
        self.filter=self.filter.cuda()
        if var is None:
            var = (dmax-dmin)/31
        self.var = var

    def expand(self, distances):
        return torch.exp(-(distances.unsqueeze(-1) - self.filter)**2 / self.var**2)

class GraphTransformer(nn.Module):
    def __init__(self, 
                 edge_dim,
                 num_class, 
                 d_model, 

                 num_heads=8,
                 dropout=0.0, 
                 max_num_nbr=24,
                 num_layers=4,
                 pre_norm=False, 
                 k_hop=3,

                 dis=32,
                 step=0.2,
                 in_embed=True, 
                 btc_size=16,
                 if_glb_q=True,
                 rangefiles=None,
                 mode='SAE',
                 factor=1.):
        super().__init__()
        self.edge_dim=edge_dim
        self.num_class=num_class
        self.d_model=d_model

        self.num_heads=num_heads
        self.dropout=dropout
        self.num_layers=num_layers
        self.in_embed=in_embed
        self.pre_norm=pre_norm
        self.if_glb_q=if_glb_q
        self.factor=factor
        if mode=='CGCNN':
            self.if_glb_q=False

        self.gd_edg=GaussianDistance(0,dis,step)
        self.num_edge_dim=self.gd_edg.filter.shape[0]
        self.max_num_nbr=max_num_nbr
    
        self.featureEng=SelfAttentionEncoder(d_model, rangefiles['node'],mode)
        self.glbFeatureEng=SelfAttentionEncoder(2*d_model, rangefiles['global'],mode)
        self.encoder = SAGATEncoder(d_model, num_heads,max_num_nbr, dropout, num_layers, edge_dim)
        self.readout=ReadOut(d_model,dropout,global_fea_dim=self.glbFeatureEng.nFea,edge_dim=edge_dim,max_nbr_num=max_num_nbr,btc_size=btc_size)
        
        self._init_optional_param()
        

    def forward(self, data, if_comp=False):
        x,_,glb_fea,nbr_fea,nbr_fea_idx,rank,degree,_,num,batch=data
        if self.if_glb_q:
            if if_comp:              
                q_star, attn_glb=self.glbFeatureEng(glb_fea.float(), if_comp)
            else:
                q_star=self.glbFeatureEng(glb_fea.float())
            setattr(self.readout,'q_star',q_star.unsqueeze(0))
            hn = self.readout.h0_fea_emb(glb_fea.float()).unsqueeze(0).to(next(self.parameters()).device)
            setattr(self.readout,'hn',self.readout.h0_ln(torch.cat(torch.chunk(hn,self.readout.num_layers,dim=-1),dim=0)))
        if if_comp:
            x_fea, attn_atom=self.featureEng(x,if_comp,num)
        else:
            x_fea=self.featureEng(x,if_comp,num)

        nbr_fea=self.gd_edg.expand(nbr_fea)
        edge_attr=self.embedding_edge(nbr_fea.float())
        
        if self.pre_norm:
            output=self.fea_bn(x_fea)
            edge_attr=self.bond_bn(edge_attr)
        output,edge_attr = self.encoder(output, 
                                edge_attr,
                                nbr_fea_idx,
                                rank,
                                degree,
                                batch,
                                self.readout,
                                self.if_glb_q)
        if self.if_glb_q:
            output=self.readout.return_output()
            self.readout.clear_attr()
        else:
            output=scatter_mean(output,batch,dim=0)
        if if_comp:
            return self.classifier(self.ff(output))*self.factor,  attn_atom, attn_glb
        return self.classifier(self.ff(output))*self.factor
    
    def _init_optional_param(self):            
        self.embedding_edge = nn.Linear(in_features=self.num_edge_dim, out_features=self.edge_dim)

        if self.pre_norm:
            self.fea_bn=nn.LayerNorm(self.d_model)
            self.bond_bn=nn.LayerNorm(self.edge_dim)
        if self.if_glb_q:
            outsize=self.readout.return_outsize()
        else:
            outsize=self.d_model
        self.ff=nn.Linear(outsize, outsize)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(outsize, self.num_class)
        )