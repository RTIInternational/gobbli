# Results: spacy-transformers
```
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/4c7c27aff9f14cc294b36e82453bb228'
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Build finished in 0.25 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1550)[0m /code/gobbli/model/spacy/model.py:176: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=1550)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Training finished in 7927.30 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation loss: 2.3743533361653136e-12
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8043999999991955
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Training loss: 2.278554696628703e-05
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/465b45ef238f480991aa09c078888cff'
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Build finished in 0.35 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Training finished in 9247.67 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation loss: 4.0018200309077656e-12
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.49799999999950195
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Training loss: 0.00012271622593980282
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/e06e1e0ba7f74b51becb654ddcfb962e'
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Build finished in 0.40 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Training finished in 7814.58 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation loss: 4.003729969781489e-12
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.49799999999950195
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Training loss: 0.00012297076051472687
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/708a1a4b6d684b96b0f597b91e1947d5'
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Build finished in 0.34 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Training finished in 5899.18 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation loss: 2.0054175209338608e-12
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8479999999991519
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:  Training loss: 3.980713727162897e-06
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/ce3aad4a5ca9499c909ac2e5883246a9'
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Build finished in 0.27 sec.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1550)[0m INFO:gobbli.experiment.base:Prediction finished in 708.64 sec.
[2m[36m(pid=1550)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1550)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                    | node_ip_address   | model_params                                 |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------------------------------------|
|  0 |  2.37435e-12 |           0.8044 |  2.27855e-05 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/4c7c27aff9f14cc294b36e82453bb228/train/f1c15f7b8f3b4789b7cc3d8509f813a4/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_bertbaseuncased_lg'}       |
|  1 |  4.00182e-12 |           0.498  |  0.000122716 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/465b45ef238f480991aa09c078888cff/train/bf4d19786f2948638cadfdcc949d5093/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_xlnetbasecased_lg'}        |
|  2 |  4.00373e-12 |           0.498  |  0.000122971 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/e06e1e0ba7f74b51becb654ddcfb962e/train/db858c5c264847478d486c850e9cd513/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_robertabase_lg'}           |
|  3 |  2.00542e-12 |           0.848  |  3.98071e-06 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/708a1a4b6d684b96b0f597b91e1947d5/train/ab68b0282a6f4c84a61d8bde5a6bcb2e/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_distilbertbaseuncased_lg'} |
```
Metrics:
--------
Weighted F1 Score: 0.8407997391662927
Weighted Precision Score: 0.8408022334815173
Weighted Recall Score: 0.8408
Accuracy: 0.8408

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](spacy-transformers/plot.png)
---