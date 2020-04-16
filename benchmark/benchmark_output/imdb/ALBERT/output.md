# Results: ALBERT
```
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/83cdde3f4ff34ce89ce6a83017a158db'
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Build finished in 0.17 sec.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Training finished in 1537.64 sec.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:  Validation loss: 0.017922103345394135
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8568
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:  Training loss: 0.0030251576885581017
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/0faea1cc752e4c6fb2b636611d87301a'
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Build finished in 0.21 sec.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Training finished in 1564.51 sec.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:  Validation loss: 0.011985438811779022
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8344
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:  Training loss: 0.01198964168727398
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/766f581291804f0480926d9ef37b5664'
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Build finished in 0.18 sec.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=696)[0m INFO:gobbli.experiment.base:Prediction finished in 135.32 sec.
[2m[36m(pid=696)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=696)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                             |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------|
|  0 |    0.0179221 |           0.8568 |   0.00302516 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/83cdde3f4ff34ce89ce6a83017a158db/train/811ffb962c834b51bf487f185c5db3a8/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v1'} |
|  1 |    0.0119854 |           0.8344 |   0.0119896  | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/0faea1cc752e4c6fb2b636611d87301a/train/1341b3c880654804bd447e8d56fad5e9/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v2'} |
```
Metrics:
--------
Weighted F1 Score: 0.8273984465860195
Weighted Precision Score: 0.8274117868243257
Weighted Recall Score: 0.8274
Accuracy: 0.8274

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](ALBERT/plot.png)
---