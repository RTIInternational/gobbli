# Results: XLNet
```
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Build finished in 0.27 sec.
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Training finished in 1087.23 sec.
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:  Validation loss: 0.03929394641598926
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.5744586831639417
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:  Training loss: 0.08142071537596918
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Build finished in 0.25 sec.
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=747)[0m INFO:gobbli.experiment.base:Prediction finished in 68.27 sec.
[2m[36m(pid=747)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=747)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                 | node_ip_address   | model_params                                                              |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:--------------------------------------------------------------------------|
|  0 |    0.0392939 |         0.574459 |    0.0814207 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/41511ad5f329428d90b391d3c37a2b16/train/089c90c152864d3cb49aa05d8f7a597c/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLNet', 'transformer_weights': 'xlnet-base-cased'} |
```
Metrics:
--------
Weighted F1 Score: 0.5061801811911584
Weighted Precision Score: 0.5029497723809235
Weighted Recall Score: 0.5342538502389803
Accuracy: 0.5342538502389803

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.39      0.09      0.15       319
           comp.graphics       0.25      0.12      0.16       389
 comp.os.ms-windows.misc       0.33      0.39      0.36       394
comp.sys.ibm.pc.hardware       0.35      0.24      0.29       392
   comp.sys.mac.hardware       0.34      0.44      0.38       385
          comp.windows.x       0.35      0.23      0.28       395
            misc.forsale       0.76      0.73      0.74       390
               rec.autos       0.57      0.68      0.62       396
         rec.motorcycles       0.53      0.43      0.47       398
      rec.sport.baseball       0.68      0.74      0.71       397
        rec.sport.hockey       0.81      0.84      0.82       399
               sci.crypt       0.46      0.70      0.56       396
         sci.electronics       0.40      0.45      0.43       393
                 sci.med       0.76      0.78      0.77       396
               sci.space       0.72      0.77      0.75       394
  soc.religion.christian       0.49      0.80      0.61       398
      talk.politics.guns       0.54      0.69      0.61       364
   talk.politics.mideast       0.60      0.84      0.70       376
      talk.politics.misc       0.49      0.44      0.46       310
      talk.religion.misc       0.00      0.00      0.00       251

                accuracy                           0.53      7532
               macro avg       0.49      0.52      0.49      7532
            weighted avg       0.50      0.53      0.51      7532


```

![Results](XLNet/plot.png)
---