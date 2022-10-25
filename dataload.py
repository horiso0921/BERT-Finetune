from torch.utils.data import Dataset, DataLoader
import pickle

class MyDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.data = xdata
        self.label = ydata
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return (x,y)

def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    xs = list(images)
    ys = list(targets)
    return xs, ys

def load_data_pkl(dir:str, mode: str, batch_size:int = 32) -> DataLoader:
    """データをロードする

    Args:
        dir (str): データがあるディレクトリ data/{dir}
        mode (str): データをロードする際に [x,y]{mode}.pklでロードするのでttvのどれか指示すること
        batch_size (int): Defaults to 32.

    Returns:
        Tuple[DataLoader]: _description_
    """

    with open(f'data/{dir}/x{mode}.pkl','br') as fr:
        xdata = pickle.load(fr)

    with open(f'data/{dir}/y{mode}.pkl','br') as fr:
        ydata = pickle.load(fr)
        
    dataset = MyDataset(xdata, ydata)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

    return dataloader