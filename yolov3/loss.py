from torch import nn
class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_noobj = 0.1
        self.lambda_obj = 1
        self.lambda_class = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):

        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        no_object_loss = self.bce(
            predictions[..., 0:1][noobj], target[..., 0:1][noobj]
        )
        
        anchors = anchors.reshape(1,3,1,1,2)
        box_pred = torch.cat(self.sigmoid(predictions[...,1:3]), torch.exp(predictions[3:5] * anchors), dim = -1)
        iou = intersection_over_union(box_pred[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), iou * target[..., 0:1][obj])    

        predictions[..., 1:3] = self.sigmoid(predictions[...,1:3])
        target[..., 3:5] = torch.log(1e-16 + target[...,3:5]/anchors)
        box_loss = self.mse(
            predictions[..., 1:5][obj], target[..., 1:5][obj]
        )

        class_loss = self.entropy(
            predictions[..., 5][obj], target[..., 5][obj].long()
        )

        loss = (
            self.lambda_obj * obj_loss +
            self.lamda_noobj * no_object_loss +
            self.lambda_class * class_loss +
            self.lambda_box * box_loss
        )
        return loss
        
