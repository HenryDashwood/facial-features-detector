import json
from io import BytesIO
import os
import requests
from typing import Union

import boto3
from fastai.vision.all import PILImage
from matplotlib import pyplot as plt
import numpy as np
from pydantic import FilePath, AnyHttpUrl
from typer import Typer

app = Typer()

client = boto3.client("sagemaker-runtime")


def process_image(path_to_image: Union[FilePath, AnyHttpUrl]):
    if os.path.isfile(path_to_image):
        img = PILImage.create(path_to_image)
        byte_array = BytesIO()
        img.save(byte_array, format="PNG")
        byte_array = byte_array.getvalue()
    else:
        byte_array = requests.get(path_to_image)._content
        img = PILImage.create(byte_array)
    return img, byte_array


def invoke(endpoint_name: str, image_path: Union[FilePath, AnyHttpUrl]):
    img, byte_array = process_image(image_path)
    img_size, _ = img.size
    response = client.invoke_endpoint(EndpointName=endpoint_name, Body=byte_array)
    normalised_results = json.loads(response["Body"].read())
    results = [r * img_size // 2 + img_size // 2 for r in normalised_results]

    return img, results


@app.command()
def predict(endpoint_name: str, image_path: str):
    img, preds = invoke(endpoint_name, image_path)
    plt.imshow(np.array(img))
    if endpoint_name == "dev-facial-features-detector":
        plt.scatter(np.array(preds[1::2]), np.array(preds[0::2]))
    else:
        plt.scatter(np.array(preds[0::2]), np.array(preds[1::2]))
    plt.show()


if __name__ == "__main__":
    app()
