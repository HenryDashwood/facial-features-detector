import boto3
from io import BytesIO
import json
import logging
import from PIL import Image
import requests
import tarfile
import torch
import torch.nn.functional as F

def load_model(bucket, key):
    s3 = boto3.resource("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    bytestream = BytesIO(obj['Body'].read())
    tar = tarfile.open(fileobj=bytestream, mode="r:gz")
    for member in tar.getmembers():
        print("Model file is :", member.name)
        f=tar.extractfile(member)
        print("Loading PyTorch model")
        model = torch.jit.load(BytesIO(f.read()), map_location=torch.device('cpu')).eval()
    return model

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class MSELossFlat(nn.MSELoss):
    def forward(self, input:Tensor, target:Tensor):
        return super().forward(input.view(-1), target.view(-1))

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

mse_loss_flat = MSELossFlat()

def input_fn(request_body):
    if isinstance(request_body, str):
        request_body = json.loads(request_body)
    img_request = requests.get(request_body['url'], stream=True)
    img = Image.open(io.BytesIO(img_request.content))
    img_tensor = img.resize(320).toTensor()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

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
    return output

model = load_model(mtpfacialfeaturesmodel)

def lambda_handler(event, context):
    input_object = input_fn(event['body'])
    response = predict_fn(input_object, model)
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }
