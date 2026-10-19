import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import VOCDATASET
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from model import Yolov1
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
from tqdm.auto import tqdm

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms #takes in a list
    def __call__(self,img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(loss_fn, dataloader, model, optimizer, device):
    loop = tqdm(dataloader, leave = True)
    mean_list = []

    for batch, (x,y) in enumerate(loop):
        x,y = x.to(device),y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())

    print(f'Mean was {sum(mean_list)/len(mean_list)}')

def main():
    model = Yolov1()
    loss_fn = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE),model, optimizer)
    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    for epoch in range(epochs):
        pred_boxes, target_boxes = getbboxes()

        mean_average_precision = mean_average_precision(pred_boxes, target_boxes)
        print(f'Mean average precision is {mean_average_precision}')

        train_fn(train_loader, model, loss_fn, optimizer, device = DEVICE)

if __name__ ==' __main__':
    main()
