from utils import load_datasets
from train_script import train
from test_script import test
from config import Config
'''
from scvi.dataset import (
                    PreFrontalCortexStarmapDataset,
                    FrontalCortexDropseqDataset,
                    RetinaDataset,
                    HematoDataset
                  )
#from scvi.models import SCANVI, VAE
#from scvi.inference import UnsupervisedTrainer, JointSemiSupervisedTrainer, SemiSupervisedTrainer
'''
import os
import json
import numpy as np
from torch import zeros, Tensor, LongTensor, log, cuda, save, load
import pdb


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

cfg = Config()

print('Loading data...')

def do_log(data):
    if isinstance(data, np.ndarray):
        data = np.log(data + 1.)
    elif isinstance(data, Tensor):
        data = log(data + 1.)
    else:
        raise TypeError('')
    return data


data = load_datasets(cfg, True, True, preprocessing=do_log)
dataloader_train = data[0]
dataloader_val = data[1]
dataloader_test = data[2]
annot_train = data[3]
annot_test = data[4]

styletransfer_test_expr = annot_test.dataset.tensors[0].cpu().numpy()
styletransfer_test_class = annot_test.dataset.tensors[1].cpu().numpy()
styletransfer_test_celltype = annot_test.dataset.tensors[2].cpu().numpy()


model = None
disc = None
#cfg.epochs = 10
epoch_dist = []

#for l1_weight in np.linspace(1e-5, 0.99, 50):
for i in range(1):
    #cfg.l1_weight = l1_weight
    model, disc = train(dataloader_train, dataloader_val, 
                        cfg, model, disc, random_seed=0)
    #save(model, '/home/mrowl/mouse_expr/source/mweights/mouse_entire.pkl')

    print('Tests...')
    model.eval()
    try:
        res = test(cfg,
            model, disc,
            annot_train,
            styletransfer_test_expr,
            styletransfer_test_class,
            styletransfer_test_celltype
        )
    except Exception as err:
        res = {
            'error': str(err)
        }
    res['architecture'] = "mmd на cat(ohe, latent)"
    #lwname = "{:.6f}".format(l1_weight)[2:]
    #os.system("mkdir /home/mrowl/mouse_expr/source/logs/vae/with_mmd")
    #with open(f'/home/mrowl/mouse_expr/source/logs/find_kernel_mu/{mmd_mu}.json', 'w') as file:
    #    json.dump(res, file, indent=4)
    os.system("rm -rf /home/mrowl/mouse_expr/source/experiment/")
    #os.system("cp /home/mrowl/mouse_expr/source/config.py /home/mrowl/mouse_expr/source/logs/avae_config.py")

    #epoch_dist.append(res)
    '''
        json.dump(res, file)
    '''
    #del model, disc
    '''
    #del styletransfer_test_expr
    #del styletransfer_test_class
    #del styletransfer_test_celltype
    del data
    del dataloader_train, dataloader_val, dataloader_test
    del annot_train, annot_test
    del scvai_genes, scvai_batches_ind, scvai_labels_ind
    cuda.empty_cache()
    '''
#with open('/home/mrowl/mouse_expr/source/epoch_dist2.json', 'w') as file:
#    json.dump(epoch_dist, file)


