# ClothesSegmentation

### Split DeepFasionMask Dataset
```PYTHONPATH=. split_deepfashionmask_data.py --data_path {data_path}```

### MlFlow Server
```mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5001```

### Train
```PYTHONPATH=. python run/train.py --config_path {config_path}```

### Test
```PYTHONPATH=. python run/test.py --config_path {config_path}```
