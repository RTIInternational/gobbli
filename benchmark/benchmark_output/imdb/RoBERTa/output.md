# Results: RoBERTa
```
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/08a2251f036a4899b6613602d62acf64'
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Build finished in 0.21 sec.
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Training finished in 1886.50 sec.
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:  Validation loss: 0.016859848795831203
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8932
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:  Training loss: 0.005073892575781792
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/dd28d9d153a74318ad848edc19d8bf5f'
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=271)[0m INFO:gobbli.experiment.base:Prediction finished in 135.27 sec.
[2m[36m(pid=271)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=271)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                            |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------|
|  0 |    0.0168598 |           0.8932 |   0.00507389 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/08a2251f036a4899b6613602d62acf64/train/e1b508abb8ed4967bedd757b43240b41/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Roberta', 'transformer_weights': 'roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8889997568650675
Weighted Precision Score: 0.8890034082922622
Weighted Recall Score: 0.889
Accuracy: 0.889

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](RoBERTa/plot.png)
---