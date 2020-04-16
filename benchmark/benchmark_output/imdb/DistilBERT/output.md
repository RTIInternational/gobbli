# Results: DistilBERT
```
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/58afa92d48d246768b760698ed6f9f8c'
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Build finished in 0.24 sec.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Training finished in 1309.59 sec.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:  Validation loss: 0.021510520422831178
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8664
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:  Training loss: 0.0006958735693711787
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/2b6000a2aa8948328277ba2ede2199e3'
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Build finished in 0.20 sec.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Training finished in 1271.24 sec.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:  Validation loss: 0.023426591290533542
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8548
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:  Training loss: 0.00114441422293894
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/519e54c2b59c4285b70585392b125613'
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Build finished in 0.23 sec.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=483)[0m INFO:gobbli.experiment.base:Prediction finished in 132.28 sec.
[2m[36m(pid=483)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=483)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                                                          |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------------------------------------|
|  0 |    0.0215105 |           0.8664 |  0.000695874 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/58afa92d48d246768b760698ed6f9f8c/train/cb6a2161339e46aca0b3f84e14ea8a54/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased'}                 |
|  1 |    0.0234266 |           0.8548 |  0.00114441  | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/2b6000a2aa8948328277ba2ede2199e3/train/c9775a474a9a472db73bca2c88249ce6/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased-distilled-squad'} |
```
Metrics:
--------
Weighted F1 Score: 0.8641388395699184
Weighted Precision Score: 0.8643870136925497
Weighted Recall Score: 0.86416
Accuracy: 0.86416

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](DistilBERT/plot.png)
---