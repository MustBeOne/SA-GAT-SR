import torch
import torch.nn.functional as F
from torch import nn
import json

class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        super(SpatialEncoding,self).__init__()
        self.max_path_distance = max_path_distance + 1
        self.bias = nn.Parameter(torch.randn(self.max_path_distance,requires_grad=True),requires_grad=True)

    def forward(self, subgraph_path_dist):
        return self.bias[subgraph_path_dist]


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim, dim_in, max_path_distance, dropout=0.5):
        super(EdgeEncoding,self).__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance+1
        self.edge_vector = nn.Embedding(self.max_path_distance, self.edge_dim)
        self.W_gate=nn.Linear(self.edge_dim+2*dim_in,edge_dim)
        self.ln = nn.LayerNorm(edge_dim)
        self.lk_relu=nn.LeakyReLU(0.1)
        self.dropout=nn.Dropout(dropout)

        nn.init.xavier_normal_(self.W_gate.weight)

    def forward(self, x, edge_attr, nbr_fea_idx, rank):
        dim=edge_attr.shape[-1]**-0.5
        edge_weight=self.edge_vector(rank)
        c_ij=edge_attr*edge_weight
        return c_ij.sum(-1)*dim


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_k, edge_dim, max_path_distance, local_hop=3, dropout=0):
        super(GraphormerAttentionHead,self).__init__()
        self.edge_dim=edge_dim
        self.local_hop=local_hop
        if edge_dim!=None:
            self.edge_encoding = EdgeEncoding(edge_dim, dim_in, max_path_distance)
        else:
            print("The gnn_type: gf-v1 doesn't use the edge attribute!")
        self.q = nn.Linear(dim_in, dim_q, bias=False)
        self.k = nn.Linear(dim_in+edge_dim, dim_k, bias=False)
        self.v = nn.Linear(dim_in+edge_dim, dim_k, bias=False)
        self.edge_emb=nn.Linear(edge_dim+dim_in*2,edge_dim)
        self.attn_dropout=nn.Dropout(dropout)
        self.global_dist_coe=nn.Embedding(5,1)
        self.lk_relu=nn.LeakyReLU(0.1)
        self.norm1 = nn.LayerNorm(dim_k)
        self.norm2 = nn.LayerNorm(dim_k)
        self.norm_edge = nn.LayerNorm(edge_dim)
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q.weight)
        nn.init.xavier_normal_(self.k.weight)
        nn.init.xavier_normal_(self.v.weight)
        nn.init.xavier_normal_(self.edge_emb.weight)

    def forward(self, x, bias, edge_attr, nbr_fea_idx, rank, degree, batch):
        c_ij=self.edge_encoding(x, edge_attr, nbr_fea_idx, rank)
        params=(x, bias, edge_attr, nbr_fea_idx, rank, degree, c_ij)
        x,edge_attr=self.global_transformer(x, bias, edge_attr, nbr_fea_idx, c_ij, rank)
        return x,edge_attr
    
    def global_transformer(self,x, bias, edge_attr, nbr_fea_idx, c_ij, rank):
        query = self.q(x)
        dim=query.shape[-1]**-0.5
        key = self.k(torch.cat((x[nbr_fea_idx],edge_attr),dim=-1))
        value = self.v(torch.cat((x[nbr_fea_idx],edge_attr),dim=-1))
        a_ij=(query.unsqueeze(1)*key).sum(-1)*dim
        attn_ij=F.softmax(a_ij,dim=-1)+F.softmax(bias,dim=-1)
        attn_ij=self.attn_dropout(torch.softmax(attn_ij,-1))
        nbr_num=edge_attr.shape[1]
        z=torch.cat([x.unsqueeze(1).repeat(1,nbr_num,1),x[nbr_fea_idx],edge_attr],dim=-1)
        edge_attr=edge_attr+self.lk_relu(self.edge_emb(z)*attn_ij.unsqueeze(-1))
        x=x+self.lk_relu((attn_ij.unsqueeze(-1)*value).sum(1))
        return self.norm2(x),self.norm_edge(edge_attr)

class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_q, dim_k, edge_dim, max_path_distance, dropout=0.2):

        super(GraphormerMultiHeadAttention,self).__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance, dropout=dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)
        self.linear_edge = nn.Linear(num_heads * edge_dim, edge_dim)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.linear_edge.weight)
        self.lk_relu=nn.LeakyReLU(0.1)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x, 
                     bias,
                     edge_attr,
                     nbr_fea_idx,
                     rank,
                     degree,
                     batch) -> torch.Tensor:

        edges,nodes=[],[]
        for attention_head in self.heads:
            node,edge=attention_head(x,
                            bias,
                            edge_attr,
                            nbr_fea_idx,
                            rank,
                            degree,
                            batch) 
            nodes.append(node)
            edges.append(edge)
        return self.lk_relu(self.linear(torch.cat(nodes, dim=-1))),\
                self.lk_relu(self.linear_edge(torch.cat(edges, dim=-1)))


class SAGATEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, max_path_distance, dropout):
        super(SAGATEncoderLayer,self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
            dropout=dropout
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(edge_dim)
        self.dropout=nn.Dropout(dropout)
        self.lk_relu=nn.LeakyReLU(0.1)

    def forward(self,input):
        x, bias, edge_attr, nbr_fea_idx, rank, degree, batch, readout, if_glb_q=input
        x_prime, edge_prime = self.attention(x, 
                                 bias,
                                 edge_attr,
                                 nbr_fea_idx,
                                 rank,
                                 degree,
                                 batch)
        x_prime = x_prime + x
        edge_prime = edge_prime + edge_attr
        x_new = self.ln_1(x_prime)
        edge_attr = self.ln_2(edge_prime)
        if if_glb_q:
            readout(x_new, edge_attr, nbr_fea_idx, rank, degree, batch)
        return (x_new, bias, edge_attr, nbr_fea_idx, rank, degree, batch, readout,if_glb_q)

class SAGATEncoder(nn.Module):
    def __init__(self,d_model, num_heads, max_num_nbr,dropout, num_layers, edge_dim):
        super(SAGATEncoder,self).__init__()
        self.r_dropout=dropout
        self.spatial_encoding = SpatialEncoding(max_path_distance=max_num_nbr)
        modules=[]
        for _ in range(num_layers):
            modules.append(SAGATEncoderLayer(
                                d_model,
                                edge_dim,
                                num_heads,
                                max_num_nbr,
                                dropout))
        self.encoder=nn.Sequential(*modules)

    def forward(self, 
                x, 
                edge_attr,
                nbr_fea_idx,
                rank,
                degree,
                batch,
                readout,
                if_glb_q):
        bias = self.spatial_encoding(rank)
        x_new, bias, edge_attr, nbr_fea_idx, rank, degree, batch, readout, if_glb_q = self.encoder((x, 
                          bias,
                          edge_attr,
                          nbr_fea_idx,
                          rank,
                          degree,
                          batch,
                          readout,
                          if_glb_q))
        return x_new,edge_attr


class SelfAttentionEncoder(nn.Module):
    def __init__(self,d_model, rangefile, mode='SAE'):
        super(SelfAttentionEncoder,self).__init__()
        self.d_model=d_model
        self.rangefile=rangefile
        self.mode=mode

        f=open(self.rangefile,'r',encoding='utf8')
        ranges=f.readlines()
        f.close()
        self.nFea=len(ranges)
        if self.mode=='SAE':
            self.attn_weights = nn.Linear(d_model,1)
            self.generaterRange()
        elif self.mode=='FFN':
            self.attn_weights = nn.Linear(self.nFea,d_model)
        elif self.mode=='CGCNN':
            f=open('./atom_init.json','r',encoding='utf8')
            initfile=json.load(f)
            self.initEmb=[]
            for i in initfile.values():
                self.initEmb.append(i)
            f.close()
            self.initEmb=torch.tensor(self.initEmb)
            self.attn_weights = nn.Linear(92,d_model)

    def forward(self, x, if_return_attn=False, num=None):
        if self.mode=='SAE':
            feas=[]
            for n,i in enumerate(self.ranges):
                indices = torch.bucketize(x[:,n], i) - 1
                indices[torch.where(indices<0)]=0
                emb= getattr(self,f'embedding_{n}')
                feas.append(emb(indices).unsqueeze(1))
            x_fea=torch.cat(feas,dim=1)
            attn=self.attn_weights(x_fea).squeeze(-1)
            attn=F.softmax(attn,dim=-1)
            x_fea=torch.einsum('ijk,ij->ik',[x_fea,attn])
            if if_return_attn:
                return x_fea,attn
            return x_fea
        elif self.mode=='FFN':
            return self.attn_weights(x)
        elif self.mode=='CGCNN':
            self.initEmb=self.initEmb.to(x.device)
            return self.attn_weights(self.initEmb[num].float())
    def generaterRange(self):
        self.ranges=[]
        f=open(self.rangefile,'r',encoding='utf8')
        ranges=f.readlines()
        f.close()
        for n,i in enumerate(ranges):
            i=i.split(' ')
            bin_edges = torch.linspace(float(i[0])-1e-4, float(i[1])+1e-4, steps=int(i[2][:-1]))
            self.ranges.append(bin_edges.cuda())
            setattr(self,f'embedding_{n}',nn.Embedding(int(i[2][:-1]),self.d_model))
        