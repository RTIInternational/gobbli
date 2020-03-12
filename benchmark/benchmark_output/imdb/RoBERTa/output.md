# Results: RoBERTa
```
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Build finished in 0.23 sec.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Training finished in 1828.11 sec.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:  Validation loss: 0.017278218799829482
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8828
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:  Training loss: 0.006217972612846643
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Build finished in 0.38 sec.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Prediction finished in 134.85 sec.
[2m[36m(pid=849)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=849)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                            |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------|
|  0 |    0.0172782 |           0.8828 |   0.00621797 | ['neg', 'pos'] | benchmark_data/model/Transformer/028e12ba75924423a3e0f545ea2184ca/train/5da16b506b084ee0b691695f9d0882b4/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Roberta', 'transformer_weights': 'roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8888282515474982
Weighted Precision Score: 0.8890044374661773
Weighted Recall Score: 0.88884
Accuracy: 0.88884

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.90      0.88      0.89     12500
         pos       0.88      0.90      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.89      0.89     25000
weighted avg       0.89      0.89      0.89     25000


```

![Results](RoBERTa/plot.png)
---