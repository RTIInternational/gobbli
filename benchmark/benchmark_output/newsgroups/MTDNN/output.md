# Results: MTDNN
```
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:MTDNN initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/MTDNN/d95a253e7914436898e245fc418b1b05'
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Build finished in 0.34 sec.
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=744)[0m /code/gobbli/model/mtdnn/model.py:204: UserWarning: MT-DNN model does not support separate validation batch size; using train batch size '16' for both training and validation.
[2m[36m(pid=744)[0m   "MT-DNN model does not support separate validation batch size; "
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Training finished in 1226.39 sec.
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:  Validation loss: 2.2677388191223145
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8753866548828988
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:  Training loss: 2.188206434249878
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:MTDNN initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/MTDNN/c91d6323aed24d60aa46d49fa236c74f'
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Build finished in 0.29 sec.
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=744)[0m INFO:gobbli.experiment.base:Prediction finished in 64.51 sec.
[2m[36m(pid=744)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=744)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                               | node_ip_address   | model_params                                          |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------|
|  0 |      2.26774 |         0.875387 |      2.18821 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/MTDNN/d95a253e7914436898e245fc418b1b05/train/26a64af73db04e1b900a0c6a6fb75269/output/model_4.pt | 172.80.10.2       | {'max_seq_length': 128, 'mtdnn_model': 'mt-dnn-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.806590411063018
Weighted Precision Score: 0.8085680053148488
Weighted Recall Score: 0.806558682952735
Accuracy: 0.806558682952735

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.00      0.00      0.00         0
           comp.graphics       0.00      0.00      0.00         0
 comp.os.ms-windows.misc       0.00      0.00      0.00         0
comp.sys.ibm.pc.hardware       0.00      0.00      0.00         0
   comp.sys.mac.hardware       0.00      0.00      0.00         0
          comp.windows.x       0.00      0.00      0.00         0
            misc.forsale       0.00      0.00      0.00         0
               rec.autos       0.00      0.00      0.00         0
         rec.motorcycles       0.00      0.00      0.00         0
      rec.sport.baseball       0.00      0.00      0.00         0
        rec.sport.hockey       0.00      0.00      0.00         0
               sci.crypt       0.00      0.00      0.00         0
         sci.electronics       0.00      0.00      0.00         0
                 sci.med       0.00      0.00      0.00         0
               sci.space       0.00      0.00      0.00         0
  soc.religion.christian       0.00      0.00      0.00         0
      talk.politics.guns       0.00      0.00      0.00         0
   talk.politics.mideast       0.00      0.00      0.00         0
      talk.politics.misc       0.00      0.00      0.00         0
      talk.religion.misc       0.00      0.00      0.00         0

               micro avg       0.00      0.00      0.00         0
               macro avg       0.00      0.00      0.00         0
            weighted avg       0.00      0.00      0.00         0


```

![Results](MTDNN/plot.png)
---