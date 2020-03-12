# Results: XLM-RoBERTa
```
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Build finished in 0.18 sec.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Training finished in 1672.08 sec.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:  Validation loss: 0.015142943879961967
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7912
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:  Training loss: 0.013316278763115407
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Build finished in 0.17 sec.
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Prediction finished in 132.19 sec.
[2m[36m(pid=1284)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1284)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0151429 |           0.7912 |    0.0133163 | ['neg', 'pos'] | benchmark_data/model/Transformer/1b57d60374a340e0a942151f161a6e83/train/aae6c048fbee4891a21fcfc18f4fc18f/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLMRoberta', 'transformer_weights': 'xlm-roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.7942145206191481
Weighted Precision Score: 0.7949246776963078
Weighted Recall Score: 0.79432
Accuracy: 0.79432

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.78      0.82      0.80     12500
         pos       0.81      0.77      0.79     12500

    accuracy                           0.79     25000
   macro avg       0.79      0.79      0.79     25000
weighted avg       0.79      0.79      0.79     25000


```

![Results](XLM-RoBERTa/plot.png)
---