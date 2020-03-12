# Results: SKLearn
```
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Training finished in 30.48 sec.
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8769160134032606
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8775961113566063
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:  Training loss: -0.9703416773166573
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1563)[0m INFO:gobbli.experiment.base:Prediction finished in 2.88 sec.
[2m[36m(pid=1563)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1563)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                             | node_ip_address   | model_params   |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------|
|  0 |    -0.876916 |         0.877596 |    -0.970342 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SKLearnClassifier/dd51ad18944540c3b7ff6289e5d39c2f/train/64cd7bfbad7a49cfb550d59cd3f8bca6/output/estimator.joblib | 172.80.10.2       | {}             |
```
Metrics:
--------
Weighted F1 Score: 0.8053276885615974
Weighted Precision Score: 0.8124855759980173
Weighted Recall Score: 0.807886351566649
Accuracy: 0.807886351566649

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.69      0.66      0.68       319
           comp.graphics       0.69      0.73      0.71       389
 comp.os.ms-windows.misc       0.73      0.73      0.73       394
comp.sys.ibm.pc.hardware       0.72      0.75      0.74       392
   comp.sys.mac.hardware       0.84      0.81      0.82       385
          comp.windows.x       0.81      0.73      0.77       395
            misc.forsale       0.74      0.86      0.80       390
               rec.autos       0.88      0.85      0.87       396
         rec.motorcycles       0.93      0.93      0.93       398
      rec.sport.baseball       0.88      0.93      0.90       397
        rec.sport.hockey       0.96      0.95      0.96       399
               sci.crypt       0.94      0.87      0.91       396
         sci.electronics       0.70      0.77      0.73       393
                 sci.med       0.86      0.83      0.84       396
               sci.space       0.88      0.91      0.90       394
  soc.religion.christian       0.74      0.90      0.81       398
      talk.politics.guns       0.70      0.88      0.78       364
   talk.politics.mideast       0.95      0.85      0.90       376
      talk.politics.misc       0.73      0.57      0.64       310
      talk.religion.misc       0.82      0.38      0.52       251

                accuracy                           0.81      7532
               macro avg       0.81      0.80      0.80      7532
            weighted avg       0.81      0.81      0.81      7532


```

![Results](SKLearn/plot.png)
---