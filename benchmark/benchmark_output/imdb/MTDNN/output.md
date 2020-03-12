# Results: MTDNN
```
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Build finished in 0.55 sec.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=207)[0m /code/gobbli/model/mtdnn/model.py:193: UserWarning: MT-DNN model does not support separate validation batch size; using train batch size '16' for both training and validation.
[2m[36m(pid=207)[0m   "MT-DNN model does not support separate validation batch size; "
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Training finished in 2652.60 sec.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:  Validation loss: 0.42352402210235596
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8866
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:  Training loss: 0.3258255124092102
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Build finished in 1.00 sec.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Prediction finished in 189.84 sec.
[2m[36m(pid=207)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=207)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                           | node_ip_address   | model_params                                          |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------|
|  0 |     0.423524 |           0.8866 |     0.325826 | ['neg', 'pos'] | benchmark_data/model/MTDNN/22da8b980a894693b8812dc521417784/train/d81d2cdf873a4b23a8f7109a2fcd2df8/output/model_4.pt | 172.80.10.2       | {'max_seq_length': 128, 'mtdnn_model': 'mt-dnn-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8820389383504452
Weighted Precision Score: 0.8820537539351417
Weighted Recall Score: 0.88204
Accuracy: 0.88204

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.88      0.88      0.88     12500
         pos       0.88      0.89      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000


```

![Results](MTDNN/plot.png)
---