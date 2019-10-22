import pickle
import torch
from sklearn.model_selection import train_test_split

from data import get_raw_data

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj

def load_weights(model, filename):
    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        def map_location(storage, loc): return storage
    else:
        map_location = None

    state_dict = torch.load(str(filename), map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)


def save_weights(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), str(filename))


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.ModuleList):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.xavier_normal_(m.weight.data)
            # m.bias.data.zero_()
            if m.bias is not None:
                m.bias.data.normal_(0, 0.01)

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


def to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(obj, (list, tuple)):
        return [to_device(o, device) for o in obj]

    if isinstance(obj, dict):
        return {k: to_device(o, device) for k, o in obj.items()}

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = obj.to(device)
    return obj

def load_datasets(cfg, test: bool = False, annot: bool = False):
    expr, class_ohe, cell_type = get_raw_data(cfg.data_dir)
    train_expression, val_expression, train_class_ohe, val_class_ohe, train_annot, test_annot = train_test_split(
        expr, class_ohe, cell_type,
        random_state=cfg.random_state,
        stratify = class_ohe.argmax(1), test_size=0.25
    )

    full_test_expr = torch.Tensor(val_expression)
    full_test_form = torch.Tensor(val_class_ohe)
    
    val_expression, test_expression, val_class_ohe, test_class_ohe = train_test_split(
        val_expression, val_class_ohe, random_state=cfg.random_state, test_size=0.15
    )
    
    val_expression_tensor = torch.Tensor(val_expression)
    val_class_ohe_tensor = torch.Tensor(val_class_ohe)
    train_expression_tensor = torch.Tensor(train_expression)
    train_class_ohe_tensor = torch.Tensor(train_class_ohe)
    test_expression_tensor = torch.Tensor(test_expression)
    test_class_ohe_tensor = torch.Tensor(test_class_ohe)
    train_annot_tensor = torch.Tensor(train_annot)
    test_annot_tensor = torch.Tensor(test_annot)
    
    if cfg.cuda and torch.cuda.is_available():
        val_expression_tensor = val_expression_tensor.cuda()
        val_class_ohe_tensor = val_class_ohe_tensor.cuda()
        train_expression_tensor = train_expression_tensor.cuda()
        train_class_ohe_tensor = train_class_ohe_tensor.cuda()
        test_expression_tensor = test_expression_tensor.cuda()
        test_class_ohe_tensor = test_class_ohe_tensor.cuda()
        full_test_expr = full_test_expr.cuda()
        train_annot_tensor = train_annot_tensor.cuda()
        test_annot_tensor = test_annot_tensor.cuda()
        full_test_form = full_test_form.cuda()
    
    trainset = torch.utils.data.TensorDataset(train_expression_tensor,
                                            train_class_ohe_tensor)
    dataloader_train = torch.utils.data.DataLoader(trainset,
                                            batch_size=cfg.batch_size,
                                            shuffle=True,
                                            #num_workers=cfg.num_workers,
                                            drop_last=True)
    valset = torch.utils.data.TensorDataset(val_expression_tensor,
                                            val_class_ohe_tensor)
    dataloader_val = torch.utils.data.DataLoader(valset,
                                                batch_size=val_expression_tensor.size(0),
                                                shuffle=False,
                                                drop_last=True)
    
    result = tuple((dataloader_train, dataloader_val))

    if test:
        testset = torch.utils.data.TensorDataset(test_expression_tensor,
                                                test_class_ohe_tensor)
        dataloader_test = torch.utils.data.DataLoader(testset,
                                                batch_size=test_expression_tensor.size(0),
                                                shuffle=False,
                                                drop_last=True)

        result += tuple((dataloader_test,))

    if annot:
        annot_dataset_train = torch.utils.data.TensorDataset(train_expression_tensor,
                                                            train_class_ohe_tensor,
                                                            train_annot_tensor)
        annot_dataloader_train = torch.utils.data.DataLoader(annot_dataset_train,
                                                            batch_size=cfg.batch_size,
                                                            shuffle=True,
                                                            drop_last=True)

        annot_dataset_test = torch.utils.data.TensorDataset(full_test_expr,
                                                            full_test_form,
                                                            test_annot_tensor)
        annot_dataloader_test = torch.utils.data.DataLoader(annot_dataset_test,
                                                            batch_size=full_test_expr.size(0),
                                                            shuffle=False,
                                                            drop_last=True)

        result += tuple((annot_dataloader_train, annot_dataloader_test))


    return result