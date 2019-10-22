from utils import load_datasets
from train_script import train
from test_script import test
from config import Config

cfg = Config()

print('Loading data...')
data = load_datasets(cfg, True, True)
dataloader_train = data[0]
dataloader_val = data[1]
dataloader_test = data[2]
annot_train = data[3]
annot_test = data[4]

print('Training...')
model, disc = train(dataloader_train, dataloader_val, cfg)

styletransfer_test_expr = annot_test.dataset.tensors[0].cpu().numpy()
styletransfer_test_class = annot_test.dataset.tensors[1].cpu().numpy()
styletransfer_test_celltype = annot_test.dataset.tensors[2].cpu().numpy()
print(styletransfer_test_celltype.shape)

print('Tests...')
test(model, disc,
        annot_train,
        styletransfer_test_expr,
        styletransfer_test_class,
        styletransfer_test_celltype
    )
