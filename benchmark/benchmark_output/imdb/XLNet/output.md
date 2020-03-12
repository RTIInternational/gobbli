# Results: XLNet
```
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Build finished in 0.67 sec.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Training finished in 2395.51 sec.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:  Validation loss: 0.017671398958563806
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.885
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:  Training loss: 0.003449625564739108
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Prediction finished in 205.30 sec.
[2m[36m(pid=710)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=710)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                              |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:--------------------------------------------------------------------------|
|  0 |    0.0176714 |            0.885 |   0.00344963 | ['neg', 'pos'] | benchmark_data/model/Transformer/7f725266c5c645ff9aae2cd91064e0b0/train/1b7e91e576da4acda9f7eb091e841dd1/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLNet', 'transformer_weights': 'xlnet-base-cased'} |
```
Metrics:
--------
Weighted F1 Score: 0.8893179724823471
Weighted Precision Score: 0.8893485289683722
Weighted Recall Score: 0.88932
Accuracy: 0.88932

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.89      0.89      0.89     12500
         pos       0.89      0.89      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.89      0.89     25000
weighted avg       0.89      0.89      0.89     25000


```

![Results](XLNet/plot.png)
---