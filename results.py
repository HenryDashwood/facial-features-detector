import json
from io import BytesIO

import boto3
from fastai.vision.all import PILImage
from typer import Typer

app = Typer()

client = boto3.client("sagemaker-runtime")


def process_image(path_to_image):
    img = PILImage.create(path_to_image)
    byte_array = BytesIO()
    img.save(byte_array, format="PNG")
    return img, byte_array.getvalue()


@app.command()
def invoke(endpoint_name: str, image_path: str):
    img, byte_array = process_image(image_path)
    response = client.invoke_endpoint(EndpointName=endpoint_name, Body=byte_array)
    normalised_results = json.loads(response["Body"].read())
    results = [r * 160 + 160 for r in normalised_results]

    return img, results


if __name__ == "__main__":
    app()
