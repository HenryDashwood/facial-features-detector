# Facial Keypoints Regression Model

### Setup

```
git clone git@github.com:WobaMedia/facial-features-detector.git
cd facial-features-detector
pip install -r requirements.txt
echo url=[MTP_API_URL] > .env
aws configure
```

### View Results

```
python results.py [ENDPOINT_NAME] [PATH_OR_URL_TO_IMAGE]
```

### Running on remote machine

In `facial_features_detector` on local machine

```
zip -r Archive.zip resized_and_user_images resized_and_user_labels.csv -x ".*" -x "__MACOSX"
scp data/Archive.zip henry@[IP_ADDRESS_OF_REMOTE_MACHINE]:/home/henry/facial-features-detector/data/
```

In `facial_features_detector` on the remote machine

```
cd data
unzip Archive.zip
```

### Training

```
aws s3 cp [PATH_TO_DATA_IN_S3] data/Archive.zip
unzip data/Archive.zip

python [MODEL_FILE] train \
  --images-path data/resized_and_user_images \
  --labels-path data/resized_and_user_labels.csv \
  --model-type efficientnet_b3a \
  --batch-size 64 \
  --frozen-epochs 5 \
  --unfrozen-epochs 0 \
  --frozen-lr 1e-2 \
  --unfrozen-lr 1e-4 \
```

### Package for Deployment

```
python [MODEL_FILE] bentoise [PATH_TO_MODEL_WEIGHTS]
```

### Test Local Deployment

```
bentoml serve [NAME]:latest
curl -X POST "http://localhost:5000/predict" -F image=@[PATH_TO_IMAGE]
```

### Deploy to Sagemaker

```
bentoml list
bentoml sagemaker deploy [NAME_OF_ENDPOINT] -b [NAME]:[TAG] --api-name predict --region us-east-1 --instance-type ml.t2.medium
```

## Results

##### efficientnet_b3a

| Frozen LR | Frozen LR | Frozen Epochs | Unfrozen Epochs | Val Loss |
| :-------: | :-------: | :-----------: | :-------------: | :------: |
|   1e-2    |   1e-4    |       5       |       50        | 0.00802  |
|   1e-2    |   1e-4    |       5       |       100       | 0.00599  |