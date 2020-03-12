# Results: RoBERTa
```
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Build finished in 0.29 sec.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Training finished in 905.13 sec.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:  Validation loss: 0.017124219072823813
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.898806893504198
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:  Training loss: 0.004850553609504876
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Build finished in 0.24 sec.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=910)[0m INFO:gobbli.experiment.base:Prediction finished in 51.53 sec.
[2m[36m(pid=910)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=910)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                 | node_ip_address   | model_params                                                            |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------|
|  0 |    0.0171242 |         0.898807 |   0.00485055 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/Transformer/64c7f40731884c05938f5ba5e96f0835/train/a5a36d238646494f91d5dbee958fa970/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Roberta', 'transformer_weights': 'roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8149290498718768
Weighted Precision Score: 0.8183203797579869
Weighted Recall Score: 0.813595326606479
Accuracy: 0.813595326606479

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.77      0.73      0.75       319
           comp.graphics       0.79      0.76      0.77       389
 comp.os.ms-windows.misc       0.80      0.72      0.76       394
comp.sys.ibm.pc.hardware       0.61      0.72      0.66       392
   comp.sys.mac.hardware       0.79      0.78      0.78       385
          comp.windows.x       0.87      0.88      0.87       395
            misc.forsale       0.87      0.87      0.87       390
               rec.autos       0.86      0.83      0.84       396
         rec.motorcycles       0.88      0.82      0.85       398
      rec.sport.baseball       0.90      0.89      0.90       397
        rec.sport.hockey       0.90      0.92      0.91       399
               sci.crypt       0.89      0.85      0.87       396
         sci.electronics       0.75      0.79      0.77       393
                 sci.med       0.88      0.93      0.91       396
               sci.space       0.92      0.89      0.91       394
  soc.religion.christian       0.87      0.80      0.84       398
      talk.politics.guns       0.69      0.79      0.73       364
   talk.politics.mideast       0.92      0.87      0.90       376
      talk.politics.misc       0.69      0.65      0.67       310
      talk.religion.misc       0.59      0.67      0.63       251

                accuracy                           0.81      7532
               macro avg       0.81      0.81      0.81      7532
            weighted avg       0.82      0.81      0.81      7532


```

![Results](RoBERTa/plot.png)
---