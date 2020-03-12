# Results: ALBERT
```
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Build finished in 0.25 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Training finished in 1534.16 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation loss: 0.018503880481421948
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8532
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Training loss: 0.00325197589895688
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Build finished in 0.38 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Training finished in 1561.44 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation loss: 0.01970496486723423
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8652
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Training loss: 0.002900374274421483
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Prediction finished in 128.00 sec.
[2m[36m(pid=208)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=208)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                             |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------|
|  0 |    0.0185039 |           0.8532 |   0.00325198 | ['neg', 'pos'] | benchmark_data/model/Transformer/d04b1868e89747efadba75150e8ce7cb/train/b96fc714aa374cf6b3e860c6dbda22f8/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v1'} |
|  1 |    0.019705  |           0.8652 |   0.00290037 | ['neg', 'pos'] | benchmark_data/model/Transformer/8ea140e7b6ff414ea7a1901498317535/train/1ea5925dece04ceeb6701fd10127c8e8/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v2'} |
```
Metrics:
--------
Weighted F1 Score: 0.8512799048191391
Weighted Precision Score: 0.8512808992791021
Weighted Recall Score: 0.85128
Accuracy: 0.85128

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.85      0.85      0.85     12500
         pos       0.85      0.85      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000


```

![Results](ALBERT/plot.png)
---