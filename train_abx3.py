# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from timeit import default_timer as timer
import transformers
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sa_gat_module.models import GraphTransformer
from sa_gat_module.data import GraphDataset,split_dataset,batch_func
from sa_gat_module.config import load_args

def train_epoch(model, loader, criterion, optimizer, epoch, use_cuda=False):
    model.train()
    running_loss = 0.0
    tic = timer()
    with tqdm(total=len(loader)) as pbar:
        pbar.set_description('Processing:')
        for i, data in enumerate(loader):
            size = len(data[1])
            if use_cuda:
                data=[i.cuda() for i in data]

            optimizer.zero_grad()
            output = model(data).squeeze(1)
            loss = criterion(output, data[1])
            loss.backward()
            optimizer.step()            
            running_loss += loss.item() * size
            pbar.update(1)
    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False, split='Val', return_result=False):
    model.eval()
    a=[]
    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0
    pred=[]
    targ=[]
    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data[1])
            if use_cuda:
                data=[i.cuda() for i in data]
            output = model(data).squeeze(1)
            a.append(data[-1][torch.where(data[1]==0)][torch.abs(output[torch.where(data[1]==0)])>1e-2])
            if return_result:
                pred.append(output)
                targ.append(data.y)
            loss = criterion(output, data[1])
            mse_loss += F.mse_loss(output, data[1]).item() * size
            mae_loss += F.l1_loss(output, data[1]).item() * size
            running_loss += loss.item() * size
    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_mse = mse_loss / n_sample
    print('{} loss: {:.4f} MSE loss: {:.4f} MAE loss: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, epoch_mse, epoch_mae, toc - tic))
    if return_result:
        return epoch_loss, epoch_mae, epoch_mse, torch.cat(pred,dim=0), torch.cat(targ,dim=0)
    return epoch_loss, epoch_mae, epoch_mse


def main(n):
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    raw_data_path = args.raw_datadirs[1]
    cache_path=args.datasetdir+args.datasets[1]+'_'+args.target[n]+'/'
    rules_path=args.rule_dirs[1]+args.target[n]+'/'
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    if not os.path.exists(rules_path):
        os.mkdir(rules_path)
    targdatas=args.targdatas
    max_num_nbr=args.max_num_nbr
    dis=args.dis

    train_set,valid_set,test_set=split_dataset(raw_data_path,args.seed,[0.9,0.05,0.05])
    train_dset = GraphDataset(cif_dir=raw_data_path,data_dir=train_set,cache_path=cache_path+'train//',max_num_nbr=max_num_nbr,target_prop_files=targdatas,dis=dis,num_workers=None,if_cache_files=True,target=args.target[n],index_file=args.index_dirs[1])
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, pin_memory=True,collate_fn=batch_func)
    
    val_dset = GraphDataset(cif_dir=raw_data_path,data_dir=valid_set,cache_path=cache_path+'val//',max_num_nbr=max_num_nbr,target_prop_files=targdatas,dis=dis,num_workers=None,if_cache_files=True,target=args.target[n],index_file=args.index_dirs[1])
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, pin_memory=True,collate_fn=batch_func)
    
    test_dset = GraphDataset(cif_dir=raw_data_path,data_dir=test_set,cache_path=cache_path+'test//',max_num_nbr=max_num_nbr,target_prop_files=targdatas,dis=dis,num_workers=None,if_cache_files=True,target=args.target[n],index_file=args.index_dirs[1])
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, pin_memory=True,collate_fn=batch_func)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, pin_memory=True,collate_fn=batch_func)
    
    model = GraphTransformer(edge_dim=args.edge_dim,
                             num_class=1,
                             d_model=args.dim_hidden,

                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             max_num_nbr=max_num_nbr,
                             pre_norm=args.prenorm,
                             k_hop=args.k_hop,

                             dis=dis,
                             step=0.2,
                             in_embed=False,
                             btc_size=args.batch_size,
                             rangefiles=args.rangefiles[1],
                             factor=args.factors[1]) 
    
    if args.use_pretrained:
        out=rules_path+'model.pth'
        state_dict=torch.load(out)
        model.load_state_dict(state_dict['model_ckpt'])
        if_return=True
        epoch_s=0
    else:
        epoch_s=0
        best_val_loss = float('inf')
        best_epoch = 0
        if_return=False
        
    if args.use_cuda:
        model.cuda()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup:
        lr_scheduler=transformers.get_polynomial_decay_schedule_with_warmup(optimizer,80,args.epochs,power=3)    

    print("Training...")
    start_time = timer()
    n=0
    train,val=[],[]
    for epoch in range(epoch_s, args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, args.use_cuda)
        val_loss, val_mae, val_mse = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')
        # test_loss, test_mae, test_mse = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Val')
        train.append(train_loss)
        val.append(val_mae)
        if args.warmup:    
            lr_scheduler.step()

        if val_mae < best_val_loss:
            n+=1
            best_val_loss = val_mae
            best_epoch = epoch
            out=rules_path+'model.pth'
            torch.save({'model_ckpt':model.state_dict()}, out)
            
    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    np.save(rules_path+'train.npy',np.array(train))
    np.save(rules_path+'val.npy',np.array(val))


if __name__ == "__main__":
    os.chdir(sys.path[0])
    main(0)
    # main(1)
    # main(2)
    # main(3)
    # main(4)
    # main(5)
    # main(6)
    # main(7)
    # main(8)