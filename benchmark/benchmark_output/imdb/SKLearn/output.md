# Results: SKLearn
```
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:SKLearnClassifier initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SKLearnClassifier/0e1cea5a30334471a64403e0a14af82b'
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Training finished in 14.64 sec.
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8831828424147392
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8832
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:  Training loss: -0.9301481917044141
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:SKLearnClassifier initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SKLearnClassifier/0437d1aa0f7f4095a6efdee1864759af'
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1125)[0m INFO:gobbli.experiment.base:Prediction finished in 7.29 sec.
[2m[36m(pid=1125)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1125)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                                 | node_ip_address   | model_params   |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------|
|  0 |    -0.883183 |           0.8832 |    -0.930148 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SKLearnClassifier/0e1cea5a30334471a64403e0a14af82b/train/94255bd9aed1427b943c6bcae9b47090/output/estimator.joblib | 172.80.10.2       | {}             |
```
Metrics:
--------
Weighted F1 Score: 0.8781994931150106
Weighted Precision Score: 0.8782062957732819
Weighted Recall Score: 0.8782
Accuracy: 0.8782

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](SKLearn/plot.png)
---