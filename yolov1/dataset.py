import torch
import os
from PIL import Image
import pandas as pd
"""
this returns image together with their bounding boxes
1. first takes annotations which contains the label like 0.0001xml
2. also take annotations which contains the image filename e.g 0.0001jpg
3. label_path(inside it contains everything i.e (class_label, x,y,w,h) which shows where exactly the position(center) of bounding box
4. we have already have the coordinates of the center of the bounding boxes, now we convert them to match the grid (SxS) i.e how they can be pos--
itioned relative to each cells presented
"""
class VOCDATASET(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, label_dir, S = 7, B = 2, C = 20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transform
        self.S = S
        self.B = B
        self.C = C
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        label_path = os.path.join([self.label_dir, self.annotations.iloc[index, 1]])
        boxes = [] 
        with open(label_path) as f:
            for lines in f.readlines():
                class_label, x,y,w,h = [float(value) if float(value) != int(float(value)) else int(value) for value in lines.replace('\n', '').split()]
                boxes.append([class_label, x, y, w, h])
        image_path = os.path.join([self.image_dir, self.annotations.iloc[index, 0]]) 
        img = Image.open(imag_path)
        boxes = torch.tensor(boxes)

        if self.transforms:
            img, boxes = self.transforms(img, boxes)

        label_matrix = torch.zeros(self.S, self.S, self.B*5 + self.C)
        for box in boxes:
            class_label, x, y, w, h = box.list()
            class_label = int(class_label)
            """
            i,j = 0,0 for every grid 
            when we calculate the below it will give the exact position of the center according to grid e.g i,j = 5,6 represents that the
            center is in row 5 and column 6
            """
            i,j = (self.S * y), (self.S * x)

            #now to calculate the X_cell and y_cell (coordinates the bounding boxes)
            x_cell, y_cell = self.S * x - j, self.S*y - i

            #now for the width of the box
            width_cell, height_cell = self.S * w, self.S * h
            """
            this is for the below conditionality
            1. we know that i and j contain the center of the object
            2. so it is responsible for the prediction of the bounding boxes
            3. provided that row, column, confidence_scores is equal to zero it means that we did not assign as it is supposed to be 1 since we know 
            i,j is the center to which the object in the grid is located
            4. 
            """
            if label_matrix[i,j, 20] == 0:
                
                label_matrix[i,j, 20] = 1

                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i,j, 21:25] = box_coordinates

                label_matrix[i,j, class_label] = 1
        return img, label_matrix
                
##understood           
        
