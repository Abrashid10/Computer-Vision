class MnistDataModule(pl.LightningModule):
    def __init__(self, batch_size,num_workers, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
    def prepare_data(self):
        datasets.MNIST(self.data_dir, download = True, train=True)
        datasets.MNIST(self.data_dir, download = True, train = False)
    def setup(self):
        entire_dataset = datasets.MNIST(
            self.data_dir,
            download = False,
            train=True,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000,10000])
        self.test_ds = datasets.MNIST(
            self.data_dir,
            download = False,
            train = False
        )
        return train_ds, val_ds, test_ds
    def train_dataloaders(self):
        train_dataloader = DataLoader(self.train_ds, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)
        return train_dataloader
    def val_dataloaders(self):
        val_dataloader = DataLoader(self.val_ds, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)
        return val_dataloader
    def test_dataloaders(self):
        test_dataloader = DataLoader(self.test_ds, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)
        return test_dataloader
