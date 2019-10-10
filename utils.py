import pickle
import torch

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
