import pandas as pd
import numpy as np
import torch
import os
from utils import intersection_over_union
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors,img_size = 416, transform = None, S = [13,26,52], C = 20):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = (anchors[0] + anchors[1] + anchors[2])
        self.img_size = img_size
        self.S = S
        self.transform = transform
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.iou_threshold = 0.5
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join([self.label_dir, self.annotations[index, 0]])
        boxes = np.roll(np.loadtxt(f = label_path, delimiter = '', ndmin = 2), 4, axis = 1) # to be [conf_score,x,y,w,h,class_label]
        img_path = os.path.join([self.img_dir, self.annotations[index, 1]])
        img = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(img, boxes)
            img = augmentations['img']
            boxes = augmentations['boxes']

        target = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        """
        targets = [
        [3,13,13,6]
        [3,26,26,6]
        [3,52,52,6]
        ]
        represents each scale output as required by the paper 
        now we have to map boxes and img to correspond to this mapping
        """
        for box in boxes:
            iou = intersection_over_union(box[2:4], self.anchors)
            anchor_indices = iou.argsort(descending = True, dim = 0)
            x, y, w, h, class_label = box
            has_anchor = [False]*3
            """
            anchor0 = for large scale (13x13) == False
            anchor1 = for medium scale (26x26) == False
            anchor2 = for small scale (52x52) == False
            """

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                """
                targets = [
                [3,13,13,6] == scale_idx = 0
                [3,26,26,6] == scale_idx = 1
                [3,52,52,6] == scale_idx = 2
                ]
                """
                anchor_scale = anchor_idx % self.num_anchors_per_scale # this returns remainder which indicate on that specific scale which anchor to use
                #now to convert according to rows and columns for the scale to which we have got
                S = self.anchors[scale_idx]
                i, j = int(S*y), int(S*x) # find where the object center is by calculating the row and column of it in the scale e.g object center is found on 3 row 2 column
                anchor_taken = target[scale_idx][anchor_scale, i, j, 0]
                if not anchor_taken and has_anchor[scale_idx]:
                    target[scale_idx][anchor_scale, i, j, 0] = 1
                    x_cell , y_cell = S*x -j, S*y - i
                    w_cell, h_cell = S*w, S*h
                    box_cordinates = torch.tensor(x_cell, y_cell, w_cell, h_cell)
                    target[scale_idx][anchor_scale, i, j, 1:5] = box_cordinates
                    
                    target[scale_idx][anchor_scale, i, j, 5] = int(class_label) 

                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou[anchor_idx] > self.iou_threshold:
                    target[scale_idx][anchor_scale, i, j, 0] = -1

        return img, tuple(target)
        
"""
so the idea is that we have our dataset already arranged in order of conf_score,x,y,w,h, class_label but the model expects it's form inorder to train with this
so, we have to convert these true labels to correspond to the model output that is
1.firstly, we initiate the target to be as,
 targets = [
        [3,13,13,6]
        [3,26,26,6]
        [3,52,52,6]
        ]
 =>  3 - represents three anchor boxes per every scale, 13 - the scale itself, 6 - contains the dataset 
 
 2. we loop through all the boxes in the dataset, then we find intersection between every box and anchors so as to get anchors that could be used for bounding boxes
 we arrange from the largest to the smallest ===== everything is explained in the code
 3. 
 
"""
        
