# Facial Keypoints Regression Model

### TODOs

- Deploy
- Experiments
- - Does unfreezing and more training help?
- - Are bigger models better / slower?

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
