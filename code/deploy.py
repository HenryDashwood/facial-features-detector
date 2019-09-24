import io
import json
import requests
from fastai.vision import Path, ImageDataBunch, cnn_learner, models, Tensor, Flatten, open_image
import torch.nn as nn
from torch.nn.functional import mse_loss

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class MSELossFlat(nn.MSELoss):
    def forward(self, input:Tensor, target:Tensor):
        return super().forward(input.view(-1), target.view(-1)) 
    
mse_loss_flat = MSELossFlat()

head_reg = nn.Sequential(
    Flatten(), 
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(51200, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, 22),
    Reshape(-1,11,2),
    nn.Tanh()
)

def model_fn(model_file):
#     path = Path('..')
    empty_data = ImageDataBunch.load_empty('../data')    
    learn = cnn_learner(
        empty_data, 
        models.resnet34,
        loss_func=mse_loss_flat,
        custom_head=head_reg
    ).to_fp16()  
    learn.path = Path('..')
    learn.load(model_file)
    return learn

def input_fn(request_body):
    img_request = requests.get(request_body, stream=True)
    img = open_image(io.BytesIO(img_request.content))
    img = img.resize(320)
    return img

def predict_fn(input_object, model):
    preds = model.predict(input_object)
    return preds

def output_fn(prediction):        
    return json.dumps(prediction)