# Results: XLM-RoBERTa
```
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Build finished in 0.28 sec.
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Training finished in 778.99 sec.
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:  Validation loss: 0.012482696061791754
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9045514803358374
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:  Training loss: 0.004303087999276819
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Build finished in 0.23 sec.
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1392)[0m INFO:gobbli.experiment.base:Prediction finished in 51.64 sec.
[2m[36m(pid=1392)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1392)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0124827 |         0.904551 |   0.00430309 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/b181a4f34e344e0f97f425bf77523ce0/train/6827e4b29d0f47c5ae417c34e9192670/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLMRoberta', 'transformer_weights': 'xlm-roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.797694912517601
Weighted Precision Score: 0.8038032723421246
Weighted Recall Score: 0.7966011683483802
Accuracy: 0.7966011683483802

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.84      0.64      0.73       319
           comp.graphics       0.77      0.81      0.79       389
 comp.os.ms-windows.misc       0.72      0.82      0.77       394
comp.sys.ibm.pc.hardware       0.67      0.73      0.70       392
   comp.sys.mac.hardware       0.86      0.73      0.79       385
          comp.windows.x       0.90      0.80      0.84       395
            misc.forsale       0.86      0.89      0.87       390
               rec.autos       0.85      0.83      0.84       396
         rec.motorcycles       0.79      0.80      0.80       398
      rec.sport.baseball       0.77      0.87      0.82       397
        rec.sport.hockey       0.93      0.85      0.89       399
               sci.crypt       0.90      0.87      0.89       396
         sci.electronics       0.73      0.77      0.75       393
                 sci.med       0.95      0.84      0.89       396
               sci.space       0.86      0.87      0.86       394
  soc.religion.christian       0.80      0.83      0.82       398
      talk.politics.guns       0.65      0.78      0.71       364
   talk.politics.mideast       0.92      0.87      0.89       376
      talk.politics.misc       0.66      0.54      0.59       310
      talk.religion.misc       0.52      0.63      0.57       251

                accuracy                           0.80      7532
               macro avg       0.80      0.79      0.79      7532
            weighted avg       0.80      0.80      0.80      7532


```

![Results](XLM-RoBERTa/plot.png)
---