# -*- coding: utf-8 -*-
import torch
import json
import random
import time
import os
import multiprocessing

from .dataset_func import *

def split_dataset(files_dir,randomseed,split_ratio):
    cif_files=list(filter(lambda x:x.endswith('.cif'),os.listdir(files_dir)))
    random.seed(randomseed)
    random.shuffle(cif_files)
    split_ratio=[int(i*len(cif_files)) for i in split_ratio]
    train_set=cif_files[:split_ratio[0]]
    valid_set=cif_files[split_ratio[0]:split_ratio[0]+split_ratio[1]]
    test_set=cif_files[split_ratio[0]+split_ratio[1]:split_ratio[0]+split_ratio[1]+split_ratio[2]]
    return train_set,valid_set,test_set

class GraphDataset(object):
    def __init__(self, dis=8,cif_dir=None,data_dir=[],target_prop_files=None,
                 cache_path=None, max_num_nbr=24, num_workers=None, if_cache_files=False,
                 if_return_id=False, target=None,**kwargs):
        self.abs_pe_list = None
        self.target=target
        self.max_num_nbr = max_num_nbr
        self.dis = dis
        self.cache_path = cache_path
        self.num_workers=num_workers
        self.abs_pe_list=None
        self.data_dir=data_dir
        self.data_dir=[cif_dir+i for i in self.data_dir]
        self.if_return_id=if_return_id
        with open(target_prop_files,'r',encoding='utf8') as f:
            self.target_prop=json.load(f)
        if if_cache_files and not os.path.exists(cache_path):
            os.mkdir(cache_path)
            self.extract_subgraphs(kwargs)
        self.cache_dir=os.listdir(os.path.dirname(self.cache_path))

    def extract_subgraphs(self,kwargs):
        print("Extracting subgraphs...")
        if self.num_workers!=None:
            pool = multiprocessing.Pool(processes = self.num_workers)
            k, m = divmod(len(self.data_dir), self.num_workers)
            work_list=[self.data_dir[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.num_workers)]
            a=0
            for n,i in enumerate(work_list):
                work_list[n]=(i,self.target_prop,self.cache_path,self.target,self.dis,self.max_num_nbr,a)
                a+=len(i)
            t=time.time()
            pool.map(cache_subgraph, work_list)
            pool.close()
            pool.join()
            print("Trainset has been finished! Take "+str(time.time()-t)+' s')
        else:
            cache_subgraph((self.data_dir,self.target_prop,self.cache_path,self.target,self.dis,self.max_num_nbr,0,kwargs))
        print("Done!")

    def __len__(self):
        return len(self.cache_dir)
    def __getitem__(self, index):
        filepath = "{}//{}".format(os.path.dirname(self.cache_path), self.cache_dir[index])
        cache_file = torch.load(filepath)
        if self.if_return_id:
            return (cache_file['x'],cache_file['y'],cache_file['glb_fea'],cache_file['nbr_fea'],
                cache_file['nbr_fea_idx'],cache_file['rank'],cache_file['degree'],cache_file['lab'],cache_file['number'],self.cache_dir[index])
        else:
            return (cache_file['x'],cache_file['y'],cache_file['glb_fea'],cache_file['nbr_fea'],
                cache_file['nbr_fea_idx'],cache_file['rank'],cache_file['degree'],cache_file['lab'],cache_file['number'])
def cache_subgraph(work_list):
    dataset, target_prop, cache_path, target, dis, max_num_nbr, ind, kwargs=work_list
    degree_max=-1
    for i,graph in enumerate(dataset):
        if cache_path is not None:
            filepath = "{}{}.pt".format(cache_path, os.path.basename(graph)[:-4])
            if os.path.exists(filepath):
                ind+=1
                continue
        if kwargs['if_abo3_system']:
            x,y,glb_fea,nbr_fea,nbr_fea_idx,rank,lab,number,if_save=create_graph_from_cif_abo3(graph,target_prop,target,dis,max_num_nbr,kwargs)
        else:
            x,y,glb_fea,nbr_fea,nbr_fea_idx,rank,lab,number,if_save=create_graph_from_cif_abx3(graph,target_prop,target,dis,max_num_nbr,kwargs)
        if not if_save:
            continue
        ind+=1
        degree=(rank==0).sum(dim=1)
        degree_max=degree_max if int((rank==0).sum(dim=1).max())<degree_max else int((rank==0).sum(dim=1).max())
        torch.save({
            'x':x,
            'y':y,
            'glb_fea': glb_fea,
            'nbr_fea': nbr_fea,
            'nbr_fea_idx': nbr_fea_idx,
            'rank': rank,
            'lab':lab,
            'degree': degree,
            'number':number
        }, filepath)
    return degree_max
def batch_func(data):
    x=torch.cat([i[0] for i in data])
    y=torch.tensor([i[1] for i in data])
    glb_fea=torch.cat([i[2].unsqueeze(0) for i in data])
    nbr_fea=torch.cat([i[3] for i in data])
    n_shift=0
    new_idx=[]
    for i in data:
        num_nodes=i[4].shape[0]
        new_idx.append(i[4]+n_shift)
        n_shift+=num_nodes
    nbr_fea_idx=torch.cat([i for i in new_idx])
    rank=torch.cat([i[5] for i in data])
    degree=torch.cat([i[6] for i in data])
    lab=torch.cat([i[7] for i in data])
    num=torch.cat([i[8] for i in data])
    batch=[torch.tensor(len(i[0])*[n]) for n,i in enumerate(data)]
    if len(data[0])==10:
        ids=[i[-1] for i in data]
        return x,y,glb_fea,nbr_fea,nbr_fea_idx,rank,degree,lab,num,torch.cat(batch),ids
    return x,y,glb_fea,nbr_fea,nbr_fea_idx,rank,degree,lab,num,torch.cat(batch)