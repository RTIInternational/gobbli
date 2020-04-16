# Results: BERT
```
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/b4a974d0cbf84a26a13d943f1313a663'
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Build finished in 0.31 sec.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Training finished in 1697.72 sec.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Validation loss: 0.5770399
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.883
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Training loss: 0.5787165
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/947ec7f4d6c648afabc639aff4d44732'
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Build finished in 0.28 sec.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Training finished in 1692.98 sec.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Validation loss: 0.63589144
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8734
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Training loss: 0.6371434
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/7280affc90ee4d18aeea25fabe3808eb'
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Build finished in 0.34 sec.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Training finished in 1699.80 sec.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Validation loss: 0.7240024
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.854
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:  Training loss: 0.7260194
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/b9b55a3e2a224e3993d9fbeab85a6833'
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Build finished in 0.40 sec.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=3321)[0m INFO:gobbli.experiment.base:Prediction finished in 217.82 sec.
[2m[36m(pid=3321)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=3321)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                                   | node_ip_address   | model_params                                               |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-----------------------------------------------------------|
|  0 |     0.57704  |           0.883  |     0.578716 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/b4a974d0cbf84a26a13d943f1313a663/train/de9b0777535341e1abbdc6b54f6c497a/output/model.ckpt-3125 | 172.80.10.2       | {'bert_model': 'bert-base-uncased', 'max_seq_length': 128} |
|  1 |     0.635891 |           0.8734 |     0.637143 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/947ec7f4d6c648afabc639aff4d44732/train/db4df22fa3e6451abd76964dfdab0cae/output/model.ckpt-3125 | 172.80.10.2       | {'bert_model': 'bert-base-cased', 'max_seq_length': 128}   |
|  2 |     0.724002 |           0.854  |     0.726019 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/7280affc90ee4d18aeea25fabe3808eb/train/bad61cd577a44744a48787586da65c9c/output/model.ckpt-3125 | 172.80.10.2       | {'bert_model': 'scibert-uncased', 'max_seq_length': 128}   |
```
Metrics:
--------
Weighted F1 Score: 0.8819691767698491
Weighted Precision Score: 0.8823994477328716
Weighted Recall Score: 0.882
Accuracy: 0.882

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](BERT/plot.png)
---