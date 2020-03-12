# Results: ALBERT
```
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Build finished in 0.26 sec.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Training finished in 704.61 sec.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:  Validation loss: 0.02395984411924432
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7808219178082192
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:  Training loss: 0.01949346131540085
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Training finished in 728.02 sec.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:  Validation loss: 0.044272430797514535
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.5413168360583297
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:  Training loss: 0.03760370841646389
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Build finished in 0.26 sec.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1232)[0m INFO:gobbli.experiment.base:Prediction finished in 44.33 sec.
[2m[36m(pid=1232)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1232)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                 | node_ip_address   | model_params                                                             |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------|
|  0 |    0.0239598 |         0.780822 |    0.0194935 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/385c6b1031dd402098b3c88d06679b85/train/e9aee3833a3c4524b5b1ff1f696ffb7e/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v1'} |
|  1 |    0.0442724 |         0.541317 |    0.0376037 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/ee45a1d7586448ad8516ad8ff31df08e/train/99cde286f0f64ee1b450e4c0e4c09dff/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v2'} |
```
Metrics:
--------
Weighted F1 Score: 0.7225559117331114
Weighted Precision Score: 0.7280078037154616
Weighted Recall Score: 0.7207912904938927
Accuracy: 0.7207912904938927

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.67      0.61      0.64       319
           comp.graphics       0.63      0.60      0.62       389
 comp.os.ms-windows.misc       0.73      0.72      0.73       394
comp.sys.ibm.pc.hardware       0.41      0.37      0.39       392
   comp.sys.mac.hardware       0.49      0.62      0.55       385
          comp.windows.x       0.87      0.75      0.80       395
            misc.forsale       0.88      0.84      0.86       390
               rec.autos       0.70      0.68      0.69       396
         rec.motorcycles       0.69      0.63      0.66       398
      rec.sport.baseball       0.87      0.76      0.81       397
        rec.sport.hockey       0.88      0.85      0.86       399
               sci.crypt       0.82      0.84      0.83       396
         sci.electronics       0.57      0.73      0.64       393
                 sci.med       0.86      0.87      0.86       396
               sci.space       0.86      0.88      0.87       394
  soc.religion.christian       0.88      0.86      0.87       398
      talk.politics.guns       0.63      0.72      0.67       364
   talk.politics.mideast       0.88      0.86      0.87       376
      talk.politics.misc       0.57      0.55      0.56       310
      talk.religion.misc       0.53      0.58      0.55       251

                accuracy                           0.72      7532
               macro avg       0.72      0.72      0.72      7532
            weighted avg       0.73      0.72      0.72      7532


```

![Results](ALBERT/plot.png)
---