# Results: XLM
```
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/7639ffbb39824aa28957eed8d4d9a415'
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.21 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 3092.26 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.02176959261894226
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.4988
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.04383168312013149
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5fb64087d96240efbb34656165bb0fb0'
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.21 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 1831.32 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.025225455497577785
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.831
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.008801721643656493
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/98ca5aa511804431bb75586f5c439618'
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.17 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Prediction finished in 232.73 sec.
[2m[36m(pid=47)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=47)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0217696 |           0.4988 |   0.0438317  | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/7639ffbb39824aa28957eed8d4d9a415/train/1325c2f951274b08b14f883c3f03707a/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-mlm-tlm-xnli15-1024'} |
|  1 |    0.0252255 |           0.831  |   0.00880172 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5fb64087d96240efbb34656165bb0fb0/train/a29d53b00ad84d26aa2e2bd6915d1fcf/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-clm-ende-1024'}       |
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

![Results](XLM/plot.png)
---