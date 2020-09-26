# Facial Keypoints Regression Model

### TODOs

- Deploy
- Experiments
- - Does unfreezing and more training help?
- - Are bigger models better / slower?

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
bentoml sagemaker deploy [NAME_OF_ENDPOINT] -b [NAME:TAG] --api-name predict --region us-east-1 --instance-type ml.t2.medium
```
