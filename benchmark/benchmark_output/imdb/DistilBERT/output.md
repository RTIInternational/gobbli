# Results: DistilBERT
```
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.20 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 1138.47 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.01951769427470863
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8794
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.0008909900698810815
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.42 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 1309.98 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.022120918272435664
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8656
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.0011140721491072327
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.45 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Prediction finished in 135.92 sec.
[2m[36m(pid=47)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=47)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                                          |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------------------------------------|
|  0 |    0.0195177 |           0.8794 |   0.00089099 | ['neg', 'pos'] | benchmark_data/model/Transformer/435c67bafd9e46c58ea9a0b422237ff3/train/9e63350a7b5d4d70aa3ba59db2169ebf/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased'}                 |
|  1 |    0.0221209 |           0.8656 |   0.00111407 | ['neg', 'pos'] | benchmark_data/model/Transformer/a08680cf48814518a3482ede744a5f93/train/725dccf1633846598aa884efeb9132d7/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased-distilled-squad'} |
```
Metrics:
--------
Weighted F1 Score: 0.8688310072637285
Weighted Precision Score: 0.8689411760669009
Weighted Recall Score: 0.86884
Accuracy: 0.86884

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.88      0.86      0.87     12500
         pos       0.86      0.88      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000


```

![Results](DistilBERT/plot.png)
---