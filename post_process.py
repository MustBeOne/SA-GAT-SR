# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
from pymatgen.core import Structure
from torch.utils.data import DataLoader
from sa_gat_module.models import GraphTransformer
from sa_gat_module.data import GraphDataset,split_dataset,batch_func
from sa_gat_module.config import load_args


def main(nn, system_n=0, if_get_attn=False, if_get_pred=False, if_get_name=False):
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw_data_path = args.raw_datadirs[system_n]
    cache_path=args.datasetdir+args.datasets[system_n]+'_'+args.target[nn]+'/'
    targdatas=args.targdatas
    rules_path=args.rule_dirs[system_n]+args.target[nn]+'/'
    
    max_num_nbr=args.max_num_nbr
    dis=args.dis

    train_set,valid_set,test_set=split_dataset(raw_data_path,args.seed,[0.8,0.1,0.1])
    train_dset = GraphDataset(cif_dir=raw_data_path,data_dir=train_set,cache_path=cache_path+'train//',max_num_nbr=max_num_nbr,target_prop_files=targdatas,dis=dis,num_workers=None,if_cache_files=True, if_return_id=True,target=args.target[nn],index_file=args.index_dir)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=False, pin_memory=True,collate_fn=batch_func)
    
    val_dset = GraphDataset(cif_dir=raw_data_path,data_dir=valid_set,cache_path=cache_path+'val//',max_num_nbr=max_num_nbr,target_prop_files=targdatas,dis=dis,num_workers=None,if_cache_files=True, if_return_id=True,target=args.target[nn],index_file=args.index_dir)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, pin_memory=True,collate_fn=batch_func)
    
    test_dset = GraphDataset(cif_dir=raw_data_path,data_dir=test_set,cache_path=cache_path+'test//',max_num_nbr=max_num_nbr,target_prop_files=targdatas,dis=dis,num_workers=None,if_cache_files=True, if_return_id=True,target=args.target[nn],index_file=args.index_dir)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, pin_memory=True,collate_fn=batch_func)
    if if_get_name:
        names=[]
        raw_data=args.raw_datadirs[system_n]
        def func(dset,names:list):
            for i in dset:
                name=i[-1][:-3]+'.cif'   
                ind=i[-3]         
                structure = Structure.from_file(raw_data+name)
                labs=structure.labels
                a=labs[torch.where(ind==1)[0][0].item()]
                b=labs[torch.where(ind==2)[0][0].item()]
                x=labs[torch.where(ind==0)[0][0].item()]
                names.append((a,b,x))
        func(train_dset,names)
        func(val_dset,names)
        func(test_dset,names)
        return names
    model = GraphTransformer(edge_dim=args.edge_dim,
                             num_class=1,
                             d_model=args.dim_hidden,

                             dropout=args.dropout,
                             num_heads=4,
                             num_layers=args.num_layers,
                             max_num_nbr=max_num_nbr,
                             pre_norm=args.prenorm,
                             k_hop=args.k_hop,

                             dis=dis,
                             step=0.2,
                             in_embed=False,
                             btc_size=args.batch_size,
                             rangefiles=args.rangefiles[system_n],
                             factor=args.factors[nn]) 
    
    state_dict=torch.load(rules_path+'model.pth')
    model.load_state_dict(state_dict['model_ckpt'])
        
    if args.use_cuda:
        model.cuda()
    glb_attn=[]
    pred=[]
    total_attn=[]
    ids=[]
    y=[]
    for n,data in enumerate(train_loader):
        ids.append(data[-1])
        y.append(data[1])
        if args.use_cuda:
            data=[i.cuda() for i in data[:10]]

        output, attn_atom, attn_glb = model(data, True)
        glb_attn.append(attn_glb)
        pred.append(output)
        total_attn.append(attn_atom)
    for n,data in enumerate(val_loader):
        ids.append(data[-1])
        y.append(data[1])
        if args.use_cuda:
            data=[i.cuda() for i in data[:10]]

        output, attn_atom, attn_glb = model(data, True)
        glb_attn.append(attn_glb)
        pred.append(output)
        total_attn.append(attn_atom)
    for n,data in enumerate(test_loader):
        ids.append(data[-1])
        y.append(data[1])
        if args.use_cuda:
            data=[i.cuda() for i in data[:10]]

        output, attn_atom, attn_glb = model(data, True)
        glb_attn.append(attn_glb)
        pred.append(output)
        total_attn.append(attn_atom)
    glb_attn=torch.cat(glb_attn)
    if if_get_attn:
        return glb_attn.mean(dim=0)
    pred=torch.cat(pred).cpu()
    y=torch.cat(y).unsqueeze(1)
    mae=list((pred-y).abs().squeeze(1).detach().numpy())
    sort=list(torch.argsort((pred-y).abs().squeeze(1)).numpy())
    sortid,sortmea=[],[]
    for i in sort:
        sortid.append(ids[i])
        sortmea.append(mae[i])
    total_attn=torch.cat(total_attn)
    print((y-pred).abs().mean().item())
    if if_get_pred:
        return pred.squeeze(1).detach().numpy()[:len(train_dset)],\
            y.squeeze(1).detach().numpy()[:len(train_dset)],\
            pred.squeeze(1).detach().numpy()[len(train_dset):],\
            y.squeeze(1).detach().numpy()[len(train_dset):],

    glb_attn=torch.sort(glb_attn.mean(dim=0))[-1]
    total_attn=torch.sort(total_attn.mean(dim=0))[-1]
    f=open(args.rule_dirs[system_n]+args.target[nn]+'/'+'attn.txt','w+',encoding='utf8')
    f.write('glb-site attn:')
    a=list(glb_attn.cpu().numpy())
    a=[str(i) for i in a]
    f.write(' '.join(a))
    f.write('\n')
    f.write('pred:\n')
    a=list(pred.squeeze(1).detach().cpu().numpy())
    preds=[]
    for i in range(len(a)):
        preds.append(str(a[i])+' '+ids[i][0])
    f.write('\n'.join(preds))
    f.write('\n')
    f.close()

def get_system_glb_attn(m,system_n, dir):
    attn=[]
    for i in range(m):
        attn.append(main(i,system_n,True))
    f=open(dir, 'w+', encoding='utf8')
    for i in attn:
        f.write(' '.join(list(i.cpu().detach().numpy().astype(str))))
        f.write('\n')
    f.close()

if __name__ == "__main__":
    os.chdir(sys.path[0])
    system_n=2
    main(0,system_n)
    # main(1,system_n)
    # main(2,system_n)
    # main(3,system_n)
    # main(4,system_n)
    # main(5,system_n)
    # main(6,system_n)
    # main(7,system_n)
    # main(8,system_n)
    