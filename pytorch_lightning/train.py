from model import NN
from dataset import MnistDataModule
import config
import pytorch_lightning as pl
from callbacks import EarlyStopping, MyPrintingCallback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

torch.set_float32_matmul_precision("medium") 
if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v0")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    model = NN(input_size = config.INPUT_SIZE, num_classes = config.NUM_CLASSES)
    dm = MnistDataModule(batch_size = config.BATCH_SIZE, num_workers = config.NUM_WORKERS, data_dir = config.DATASET)
    trainer = pl.Trainer(accelerator=config.ACCELERATOR, devices=config.DEVICES, min_epochs=1, max_epochs=3, precision=config.PRECISION, callbacks = [MyPrintingCallback(), EarlyStopping(monitor = 'val_loss')])
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
