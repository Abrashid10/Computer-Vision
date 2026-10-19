import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01

def gram_matrix(feature_map):
    batch_size, channels, width, height = feature_map.shape
    features = feature_map.view(channels, width*height)
    G = torch.mm(features, features.t())
    return G

def content_loss(gen_feature, content_feature):
    return F.mse_loss(content_features, gen_feature)
    
def style_loss(gen_feature, style_feature):
    loss = 0
    for layer in gen_feature:
        g = gram_matrix(gen_feature[layer])
        s = gram_matrix(style_feature[layer])

        loss += F.mse_loss(g,s)

    return loss

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.selected_layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '28': 'conv5_1'
        }

        self.models = models.vgg19(pretrained = True)

    def forward(self, x):
        features = {}
        for layer_num, layer in enumerate(self.models.features):
            x = layer(x)
            if str(layer_num) in self.selected_layers:
                features[self.selected_layers[str(layer_num)]] = x
            return features

def load_image(image_path):
    img = Image.open(image_path)
    img = loader(img).unsqueeze(0)
    return img.to(device)

device = 'cpu' if torch.cuda.is_available() else 'cpu'

imgsize = 356

loader = transforms.Compose([
    transforms.Resize([imgsize, imgsize]),
    transforms.ToTensor()
])

original_img = load_image("annahathaway.png") #put some image here
style_img = load_image("style.jpg") #put styles here

generated = torch.randn_like(original_img, device = device, requires_grad = True)
optimizer = torch.optim.Adam([generated], lr=learning_rate)
#alternative is cloning
#generated_clone = original_img.clone().requires_grad_(True)
model = VGG().to(device)

for step in range(training_steps):
    gen_features = model(generated) #-> returns a dictionary {0: conv1_1,...} similar to below
    content_features = model(original_img)
    style_features = model(style_img)

    
    c_loss = content_loss(
        gen_features['conv4_1'],
        content_features['conv4_1']
    )
    
    s_loss = style_loss(gen_features, style_features)
    
    total_loss = alpha * c_loss + beta * s_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")
