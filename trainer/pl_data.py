import lightning as L
from torch.utils.data import DataLoader

class data_wrapper(L.LightningDataModule):
    def __init__(self, config, train_ds, val_ds):
        super().__init__()
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.bs_train, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.bs_val, num_workers=4, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.bs_test, num_workers=4, shuffle=False, drop_last=False)

    def setup(self, stage: str):
        pass
