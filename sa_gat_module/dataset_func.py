# -*- coding: utf-8 -*-
import torch
from pymatgen.core import Structure
import numpy as np
import json
import warnings
import os

def ABO3Feature(structure,ind,angl):
    const=2**0.5
    x=[]
    glb_fea=[]
    number=[]
    glb_fea.append(structure.density)
    glb_fea.append(structure.volume)
    glb_fea.append(structure.lattice.abc[0])
    glb_fea.append(structure.lattice.abc[1])
    glb_fea.append(structure.lattice.abc[2])
    oxidation_states = structure.composition.oxi_state_guesses()
    if oxidation_states==[]:
        return -1,-1,-1
    for i in structure.species:
        features=[]
        features.append(i.atomic_radius)
        features.append(i.average_ionic_radius)
        features.append(i.X)
        features.append(i.ionization_energy)
        features.append(i.electron_affinity)
        features.append(i.atomic_mass)
        features.append(oxidation_states[0][i.name])
        x.append(torch.tensor(features).unsqueeze(0))
        number.append(i.Z)
    ra=x[ind.index(1)][0][0].item()
    rb=x[ind.index(2)][0][0].item()
    ro=x[ind.index(0)][0][0].item()
    t=(ra+ro)/(rb+ro)/const
    mu=rb/ro
    mu_t=mu/t
    glb_fea.append(t)
    glb_fea.append(mu)
    glb_fea.append(mu_t)
    glb_fea.append(np.cos(np.deg2rad(angl)))
    return torch.cat(x,dim=0), torch.cat([torch.tensor(glb_fea).unsqueeze(0),x[ind.index(1)],x[ind.index(2)]],dim=1).squeeze(0), torch.tensor(number)
    
def create_graph_from_cif_abo3(cif_file,target_prop,target,dis,max_num_nbr,kwargs):   
    with open(kwargs['index_file'],'r',encoding='utf8') as f:
        ang=json.load(f)
    ind=ang[os.path.basename(cif_file)][1]
    angl=ang[os.path.basename(cif_file)][0]
    structure = Structure.from_file(cif_file)
    x, glb_fea, number=ABO3Feature(structure,ind,angl)
    if type(x) is not torch.Tensor:
        return 0,0,0,0,0,0,0,0,False
    targ=target_prop[os.path.basename(cif_file)[:-4]][target]
    if targ!= "na":
        y=torch.tensor(float(targ)) 
    else:
        return 0,0,0,0,0,0,0,0,False
    if dis==None:
        dis=structure.distance_matrix[structure.distance_matrix!=0].min()
    nn = structure.get_all_neighbors(dis,include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in nn]

    nbr_fea_idx, nbr_fea, nbr_fea_v = [], [], []
    rank=[]
    for nn,nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            warnings.warn('{} not find enough neighbors to build graph. '
                            'If it happens frequently, consider increase '
                            'radius.'.format(cif_file))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                            [dis + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                    nbr[:max_num_nbr])))
            for i in range(max_num_nbr):
                nbr_fea_v.append(structure.cart_coords[nn]-nbr[i][0].coords)
        nbr_dist,t,n=[int(i*10000) for i in nbr_fea[nn]],[0],0
        m=nbr_dist[0]
        for i in nbr_dist[1:]:
            if i>m:
                m=i
                n+=1
                t.append(n)
            else:
                t.append(n)
        rank.append(t)
    nbr_fea_idx, nbr_fea, nbr_fea_v, rank= np.array(nbr_fea_idx), np.array(nbr_fea), np.array(nbr_fea_v), np.array(rank)
    nbr_fea = torch.Tensor(nbr_fea)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    rank = torch.LongTensor(rank)
    lab=torch.LongTensor(ind)
    return x,y,glb_fea,nbr_fea,nbr_fea_idx,rank,lab,number,True


def ABX3Feature(structure,ind,angl):
    const=2**0.5
    x=[]
    glb_fea=[]
    number=[]
    glb_fea.append(structure.density)
    glb_fea.append(structure.volume)
    glb_fea.append(structure.lattice.abc[0])
    glb_fea.append(structure.lattice.abc[1])
    glb_fea.append(structure.lattice.abc[2])
    oxidation_states = structure.composition.oxi_state_guesses()
    if oxidation_states==[]:
        return -1,-1, -1
    for i in structure.species:
        features=[]
        features.append(i.atomic_radius)
        features.append(i.average_ionic_radius)
        features.append(i.X)
        features.append(i.ionization_energy)
        features.append(i.electron_affinity)
        features.append(i.atomic_mass)
        features.append(oxidation_states[0][i.name])
        x.append(torch.tensor(features).unsqueeze(0))
        number.append(i.Z)
    ra=x[ind.index(1)][0][1].item()
    rb=x[ind.index(2)][0][1].item()
    rx=x[ind.index(0)][0][1].item()
    t=(ra+rx)/(rb+rx)/const
    mu=rb/rx
    mu_t=mu/t
    glb_fea.append(t)
    glb_fea.append(mu)
    glb_fea.append(mu_t)
    glb_fea.append(np.cos(np.deg2rad(angl)))
    return torch.cat(x,dim=0), torch.cat([torch.tensor(glb_fea).unsqueeze(0),x[ind.index(1)],x[ind.index(2)],x[ind.index(0)]],dim=1).squeeze(0), torch.tensor(number)
    
def create_graph_from_cif_abx3(cif_file,target_prop,target,dis,max_num_nbr,kwargs):   
    with open(kwargs['index_file'],'r',encoding='utf8') as f:
        ang=json.load(f)
    ind=ang[os.path.basename(cif_file)][1]
    angl=ang[os.path.basename(cif_file)][0]
    structure = Structure.from_file(cif_file)
    x, glb_fea, number=ABX3Feature(structure,ind,angl)
    if type(x) is not torch.Tensor:
        return 0,0,0,0,0,0,0,0,False
    targ=target_prop[os.path.basename(cif_file)[:-4]][target]
    if targ!= "na":
        y=torch.tensor(float(targ)) 
    else:
        return 0,0,0,0,0,0,0,0,False
    if dis==None:
        dis=structure.distance_matrix[structure.distance_matrix!=0].min()
    nn = structure.get_all_neighbors(dis,include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in nn]

    nbr_fea_idx, nbr_fea, nbr_fea_v = [], [], []
    rank=[]
    for nn,nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            warnings.warn('{} not find enough neighbors to build graph. '
                            'If it happens frequently, consider increase '
                            'radius.'.format(cif_file))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                            [dis + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                    nbr[:max_num_nbr])))
            for i in range(max_num_nbr):
                nbr_fea_v.append(structure.cart_coords[nn]-nbr[i][0].coords)
        nbr_dist,t,n=[int(i*10000) for i in nbr_fea[nn]],[0],0
        m=nbr_dist[0]
        for i in nbr_dist[1:]:
            if i>m:
                m=i
                n+=1
                t.append(n)
            else:
                t.append(n)
        rank.append(t)
    nbr_fea_idx, nbr_fea, nbr_fea_v, rank= np.array(nbr_fea_idx), np.array(nbr_fea), np.array(nbr_fea_v), np.array(rank)
    nbr_fea = torch.Tensor(nbr_fea)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    rank = torch.LongTensor(rank)
    lab=torch.LongTensor(ind)
    return x,y,glb_fea,nbr_fea,nbr_fea_idx,rank,lab,number,True
