# Results: DistilBERT
```
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Build finished in 0.39 sec.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Training finished in 549.11 sec.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:  Validation loss: 0.011513017195347289
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9076447193990278
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:  Training loss: 0.0015462252528830307
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Build finished in 0.20 sec.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Training finished in 561.02 sec.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:  Validation loss: 0.011764843772456438
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9054352629253204
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:  Training loss: 0.0021321167292695075
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Build finished in 0.28 sec.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1069)[0m INFO:gobbli.experiment.base:Prediction finished in 46.49 sec.
[2m[36m(pid=1069)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1069)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                                          |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------------------------------------|
|  0 |    0.011513  |         0.907645 |   0.00154623 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/ef5c5f33ab2443c9b77278b93372615a/train/3d0947bf191b4b53bb79c6cb2cedf931/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased'}                 |
|  1 |    0.0117648 |         0.905435 |   0.00213212 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/60593132efd845029de57170a9db3fc4/train/0073405064024f72b1b4da108dd9650d/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased-distilled-squad'} |
```
Metrics:
--------
Weighted F1 Score: 0.8275780346230087
Weighted Precision Score: 0.8301557129725602
Weighted Recall Score: 0.8266064790228359
Accuracy: 0.8266064790228359

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.77      0.70      0.73       319
           comp.graphics       0.78      0.80      0.79       389
 comp.os.ms-windows.misc       0.80      0.77      0.79       394
comp.sys.ibm.pc.hardware       0.67      0.70      0.68       392
   comp.sys.mac.hardware       0.75      0.83      0.79       385
          comp.windows.x       0.90      0.84      0.87       395
            misc.forsale       0.90      0.85      0.87       390
               rec.autos       0.89      0.86      0.87       396
         rec.motorcycles       0.80      0.82      0.81       398
      rec.sport.baseball       0.94      0.89      0.91       397
        rec.sport.hockey       0.93      0.93      0.93       399
               sci.crypt       0.91      0.88      0.90       396
         sci.electronics       0.76      0.82      0.79       393
                 sci.med       0.92      0.92      0.92       396
               sci.space       0.87      0.91      0.89       394
  soc.religion.christian       0.89      0.91      0.90       398
      talk.politics.guns       0.72      0.76      0.74       364
   talk.politics.mideast       0.97      0.86      0.92       376
      talk.politics.misc       0.67      0.63      0.65       310
      talk.religion.misc       0.63      0.74      0.68       251

                accuracy                           0.83      7532
               macro avg       0.82      0.82      0.82      7532
            weighted avg       0.83      0.83      0.83      7532


```

![Results](DistilBERT/plot.png)
---