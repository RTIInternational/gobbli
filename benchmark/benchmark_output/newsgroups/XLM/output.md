# Results: XLM
```
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:Build finished in 39.63 sec.
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:Training finished in 1411.55 sec.
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:  Validation loss: 0.09424681090381469
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.047724259832081305
[2m[36m(pid=588)[0m INFO:gobbli.experiment.base:  Training loss: 0.20573352292260733
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Build finished in 0.26 sec.
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Training finished in 746.59 sec.
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:  Validation loss: 0.023487805003251747
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8616880247459126
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:  Training loss: 0.005595734619005894
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Build finished in 0.26 sec.
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=590)[0m INFO:gobbli.experiment.base:Prediction finished in 54.21 sec.
[2m[36m(pid=590)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=590)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0942468 |        0.0477243 |   0.205734   | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/6b59f0656a964c8797b50d9d57148c62/train/dd07e98ba37f4f98a0d680d7f9f1c051/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-mlm-tlm-xnli15-1024'} |
|  1 |    0.0234878 |        0.861688  |   0.00559573 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/3fbc2750fc524ea5b772a659e8acf819/train/fead5a178643416aa5e7c2c88f8337cf/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-clm-ende-1024'}       |
```
Metrics:
--------
Weighted F1 Score: 0.755015567457987
Weighted Precision Score: 0.7596497698871361
Weighted Recall Score: 0.7550451407328731
Accuracy: 0.7550451407328731

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.73      0.65      0.68       319
           comp.graphics       0.69      0.64      0.67       389
 comp.os.ms-windows.misc       0.62      0.52      0.56       394
comp.sys.ibm.pc.hardware       0.53      0.67      0.59       392
   comp.sys.mac.hardware       0.74      0.71      0.73       385
          comp.windows.x       0.74      0.86      0.79       395
            misc.forsale       0.85      0.84      0.85       390
               rec.autos       0.79      0.74      0.76       396
         rec.motorcycles       0.78      0.75      0.77       398
      rec.sport.baseball       0.76      0.80      0.78       397
        rec.sport.hockey       0.85      0.81      0.83       399
               sci.crypt       0.89      0.87      0.88       396
         sci.electronics       0.68      0.72      0.70       393
                 sci.med       0.87      0.89      0.88       396
               sci.space       0.87      0.86      0.87       394
  soc.religion.christian       0.84      0.90      0.87       398
      talk.politics.guns       0.69      0.83      0.76       364
   talk.politics.mideast       0.95      0.76      0.85       376
      talk.politics.misc       0.67      0.57      0.62       310
      talk.religion.misc       0.55      0.59      0.57       251

                accuracy                           0.76      7532
               macro avg       0.75      0.75      0.75      7532
            weighted avg       0.76      0.76      0.76      7532


```

![Results](XLM/plot.png)
---