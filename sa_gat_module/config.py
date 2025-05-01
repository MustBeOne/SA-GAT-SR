import argparse
import torch

targ=["formation_energy_peratom", "optb88vdw_bandgap", "optb88vdw_total_energy", "ehull",
      "bulk_modulus_kv", "shear_modulus_gv", "epsx", "epsy", "epsz"]
datasets=["abo3","abx3","abxo3"]
datadirs=['./rawdata/abo3/','./rawdata/abx3/','./rawdata/abxo3/']
rules=['./rules/ruleso/','./rules/rulesx/','./rules/rulesxo/']
index=['./rawdata/ango.json','./rawdata/angx.json','./rawdata/angxo.json']
ranges=[{'node':'./rules/ruleso/noderange.txt','global':'./rules/ruleso/globalrange.txt'},
        {'node':'./rules/rulesx/noderange.txt','global':'./rules/rulesx/globalrange.txt'},
        {'node':'./rules/rulesxo/noderange.txt','global':'./rules/rulesxo/globalrange.txt'}]
factors=[1.,1.,1.,1.,8.,8.,8.,8.,8.]


def load_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--datasets', type=list, default=datasets, help='name of dataset')
    parser.add_argument('--datasetdir', type=str, default="./j_datasets/")
    parser.add_argument('--target', type=str, default=targ, help='name of target property')
    parser.add_argument('--use-pretrained', type=bool, default=False)
    parser.add_argument('--pretrained-name', type=str, default='./model.pth')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    
    parser.add_argument('--num-heads', type=int, default=4, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=4, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=128, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.3 , help="dropout ratio")
    parser.add_argument('--lr', type=float, default=0.00008, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--k-hop', type=int, default=2, help="Number of hops to use when extracting subgraphs around each node")
    parser.add_argument('--max-num-nbr', type=int, default=12)
    parser.add_argument('--dis', type=int, default=12)
    parser.add_argument('--prenorm', default=True)
    parser.add_argument('--edge-dim', type=int, default=48, help='edge features hidden dim')
    parser.add_argument('--warmup', type=bool, default=True, help="if warmup")
    
    parser.add_argument('--raw-datadirs', type=list, default=datadirs)
    parser.add_argument('--raw-datadir', type=str, default='./rawdata/abxo3/')
    parser.add_argument('--rule-dir', type=str, default='./temp/')
    parser.add_argument('--rule-dirs', type=list, default=rules)
    parser.add_argument('--index-dir', type=str, default='./rawdata/angxo.json')
    parser.add_argument('--index-dirs', type=list, default=index)
    parser.add_argument('--factors', type=list, default=factors)
    parser.add_argument('--targdatas', type=str, default='./j_datasets/jarvis_dft.json')
    parser.add_argument('--rangefiles', type=list, default=ranges)

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    return args