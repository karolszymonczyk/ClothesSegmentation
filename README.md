# ClothesSegmentation

### MlFlow Server
```mlflow ui --backend-store-uri sqlite:///mlflow.db```

### Train
```PYTHONPATH=. python run/train.py --config_path {config_path}```
#### Example
```PYTHONPATH=. python run/train.py --config_path run/configs/train_config.yaml```

### Test
```PYTHONPATH=. run/test.py --config_path run/configs/test_config.yaml```
