import io
import json
import requests

from fastai.vision import Path, ImageDataBunch, cnn_learner, models, Tensor, Flatten, open_image
import torch.nn as nn
from torch.nn.functional import mse_loss
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

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
    empty_data = ImageDataBunch.load_empty('../data')    
    learn = cnn_learner(
        empty_data, 
        models.resnet34,
        loss_func=mse_loss_flat,
        custom_head=head_reg
    )
    learn.path = Path('..')
    learn.load(model_file)
    return learn

def input_url(url):
    img_request = requests.get(request_body, stream=True)
    img = open_image(io.BytesIO(img_request.content))
    img = img.resize(320)
    return img

def input_file(img_bytes):
    img = open_image(io.BytesIO(img_bytes))
    img = img.resize(320)
    return img

def predict_fn(input_object, model):
    preds = model.predict(input_object)
    return preds

def preds_to_dict(tensor):
    
    tensor = tensor.numpy()
    
    output = {
        'leftHeadPoint_x': str(tensor[0][0]), 'leftHeadPoint_y': str(tensor[0][1]),
        'leftEarPoint_x': str(tensor[1][0]), 'leftEarPoint_y': str(tensor[1][1]),
        'topHeadPoint_x': str(tensor[2][0]), 'topHeadPoint_y': str(tensor[2][1]),
        'rightEarPoint_x': str(tensor[3][0]), 'rightEarPoint_y': str(tensor[3][1]),
        'rightHeadPoint_x': str(tensor[4][0]), 'rightHeadPoint_y': str(tensor[4][1]),
        'chinPoint_x': str(tensor[5][0]), 'chinPoint_y': str(tensor[5][1]),
        'leftEyePoint_x': str(tensor[6][0]), 'leftEyePoint_y': str(tensor[6][1]),
        'rightEyePoint_x': str(tensor[7][0]), 'rightEyePoint_y': str(tensor[7][1]),
        'leftMouthPoint_x': str(tensor[8][0]), 'leftMouthPoint_y': str(tensor[8][1]),
        'centreMouthPoint_x': str(tensor[9][0]), 'centreMouthPoint_y': str(tensor[9][1]),
        'rightMouthPoint_x': str(tensor[10][0]), 'rightMouthPoint_y': str(tensor[10][1]),
    }
    
    print(output)
    
    return output

model = model_fn('fastai_resnet34')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()    
        img = input_file(img_bytes)
        preds = predict_fn(img, model)
        output = preds_to_dict(preds[1])
        return jsonify(output)

if __name__ == '__main__':
    app.run()