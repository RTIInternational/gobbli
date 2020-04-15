# Results: XLNet
```
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/bcc8a9b71b50460abf8789b013c978d5'
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Build finished in 0.16 sec.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Training finished in 2385.69 sec.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Validation loss: 0.019337966107577084
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8936
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Training loss: 0.0028089337879791854
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/b4a59e2c2e724e4481b2a139c32cfdab'
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Build finished in 0.17 sec.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Prediction finished in 204.68 sec.
[2m[36m(pid=45)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=45)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                              |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:--------------------------------------------------------------------------|
|  0 |     0.019338 |           0.8936 |   0.00280893 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/bcc8a9b71b50460abf8789b013c978d5/train/559d8078f84e4fa4b8f33f9c950110af/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLNet', 'transformer_weights': 'xlnet-base-cased'} |
```
Metrics:
--------
Weighted F1 Score: 0.8896881198666413
Weighted Precision Score: 0.8901710377196039
Weighted Recall Score: 0.88972
Accuracy: 0.88972

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](XLNet/plot.png)
---