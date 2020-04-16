# Results: XLM-RoBERTa
```
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5a35a362b75a477bbc3fa0f1d4b7b329'
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:Build finished in 0.16 sec.
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:Training finished in 1685.29 sec.
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:  Validation loss: 0.02177449142932892
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.4892
[2m[36m(pid=911)[0m INFO:gobbli.experiment.base:  Training loss: 0.021700730556249617
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/6c89be08abf54bfc983d6809e469f44e'
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Build finished in 0.23 sec.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Prediction finished in 132.11 sec.
[2m[36m(pid=910)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=910)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0217745 |           0.4892 |    0.0217007 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5a35a362b75a477bbc3fa0f1d4b7b329/train/fe8dfe640bf244ffab350176432dce17/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLMRoberta', 'transformer_weights': 'xlm-roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.33333333333333326
Weighted Precision Score: 0.25
Weighted Recall Score: 0.5
Accuracy: 0.5

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](XLM-RoBERTa/plot.png)
---