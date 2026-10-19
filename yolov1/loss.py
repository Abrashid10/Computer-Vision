#loss function implementation
import torch
from torch import nn
from utils import intersection_over_union
class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super().__init__()
        self.mse = nn.MSE(reduction = 'sum')
        self.S = S
        self.B = B
        self.C = C

        self.noobj = 0.5
        self.obj = 5
    def forward(self, predictions, target):
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)

        iou_max, best_box = ious.max(dim = 0)
        # this returns (0,1) as indices 
        box_exists = target[..., 20].unsqueeze(3) #this returns 1obj in the paper of shape (batch_size, 7,7,1)
        
        #box_loss
        box_predictions = (
            box_exists * (best_box*predictions[..., 26:30]) # this is the same as (batch_size,7,7,1) x (batch_size,7,7,4)
            + ((1-best_box) * predictions[..., 21:25])
        )
        box_targets = box_exists * target[..., 21:25]
        

        box_predictions[..., 2:4] = torch.sign(box_predictions[...,2:4]*torch.sqrt(torch.abs(box_predictions[..., 2:4]+1e-6))) # same as (batch_size,7,7,2)
        box_targets[..., 2:4] = torch.sqrt(box_targets[...,2:4])

        box_loss = self.mse(
            nn.Flatten(box_predictions, end_dim=-2),#(batch_size*7*7,4)
            nn.Flatten(box_targets, end_dim=-2)
        )
        #object_loss or confidence score loss termed as Ci in the paper
        pred_box = (
            best_box*predictions[..., 25:26]+(1-best_box)*predictions[..., 20:21] # (batch_size, 7,7,1)
        )
        object_loss = self.mse(
            nn.Flatten(pred_box),# same as (batch_size*7*7*1)
            nn.Flatten(box_targets[20:21])
        )
        #no object loss
        #for the first bounding box
        no_object_loss = self.mse(
            torch.flatten((1 - box_exists) * predictions[..., 20:21], start_dim=1), #(batch_size, 7*7*1)
            torch.flatten((1 - box_exists) * target[..., 20:21], start_dim=1),
        #for the second bounding box
        no_object_loss += self.mse(
            torch.flatten((1-box_exists)*predictions[...,25:26],start_dim = 1),#(batch_size, 7*7*1)
            torch.flatten((1-box_exists)*target[...,20:21], start_dim = 1)
        )
        #class loss
        class_loss = self.mse(
            torch.flatten(box_exists * predictions[..., :20], end_dim = -2)#(batch_size*7*7,1)
            torch.flatten(box_exists * target[..., :20], end_dim = -2)
        )

        loss = (
            self.obj*box_loss + self.obj*object_loss
            + self.noobj*no_object_loss + class_loss
        )
        return loss
