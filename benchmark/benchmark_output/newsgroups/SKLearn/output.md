# Results: SKLearn
```
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:SKLearnClassifier initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SKLearnClassifier/32b966f0016e4f6fa6aa9966a8c6fd81'
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Training finished in 34.51 sec.
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8812004917087203
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8815731330092798
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:  Training loss: -0.9690307665707693
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:SKLearnClassifier initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SKLearnClassifier/42f3b187edf44a78b04c33188f66de0b'
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=2629)[0m INFO:gobbli.experiment.base:Prediction finished in 4.30 sec.
[2m[36m(pid=2629)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=2629)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                                 | node_ip_address   | model_params   |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------|
|  0 |      -0.8812 |         0.881573 |    -0.969031 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SKLearnClassifier/32b966f0016e4f6fa6aa9966a8c6fd81/train/327d971e8f1e448fad3890a03883816e/output/estimator.joblib | 172.80.10.2       | {}             |
```
Metrics:
--------
Weighted F1 Score: 0.8067615329675221
Weighted Precision Score: 0.8130511614151636
Weighted Recall Score: 0.8093467870419543
Accuracy: 0.8093467870419543

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

![Results](SKLearn/plot.png)
---