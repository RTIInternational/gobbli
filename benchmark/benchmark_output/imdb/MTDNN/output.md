# Results: MTDNN
```
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:MTDNN initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/MTDNN/97bd914c45964562810bc65fbb005958'
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Build finished in 0.27 sec.
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3537)[0m /code/gobbli/model/mtdnn/model.py:204: UserWarning: MT-DNN model does not support separate validation batch size; using train batch size '16' for both training and validation.
[2m[36m(pid=3537)[0m   "MT-DNN model does not support separate validation batch size; "
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Training finished in 2691.56 sec.
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:  Validation loss: 0.4333544969558716
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8782
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:  Training loss: 0.3258266746997833
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:MTDNN initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/MTDNN/2400ab65a9b1404a893cb9e3e6c18f0b'
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Build finished in 0.31 sec.
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=3537)[0m INFO:gobbli.experiment.base:Prediction finished in 187.28 sec.
[2m[36m(pid=3537)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=3537)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels         | checkpoint                                                                                                                                               | node_ip_address   | model_params                                          |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------|
|  0 |     0.433354 |           0.8782 |     0.325827 | False        | ['neg', 'pos'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/MTDNN/97bd914c45964562810bc65fbb005958/train/a64ab92d26334450b788b96d243fba71/output/model_4.pt | 172.80.10.2       | {'max_seq_length': 128, 'mtdnn_model': 'mt-dnn-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8816735903057176
Weighted Precision Score: 0.8817627197352919
Weighted Recall Score: 0.88168
Accuracy: 0.88168

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.00      0.00      0.00         0
         pos       0.00      0.00      0.00         0

   micro avg       0.00      0.00      0.00         0
   macro avg       0.00      0.00      0.00         0
weighted avg       0.00      0.00      0.00         0


```

![Results](MTDNN/plot.png)
---