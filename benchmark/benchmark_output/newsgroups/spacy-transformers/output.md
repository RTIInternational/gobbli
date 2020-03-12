# Results: spacy-transformers
```
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 100.49 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m /code/gobbli/model/spacy/model.py:174: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=47)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 4113.50 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 1.646793939981946e-11
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8873177198369981
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 6.1027339446522e-06
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 108.29 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 5015.53 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 2.0434902947377245e-11
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.864339372510542
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 1.632588307021673e-05
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 150.97 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 4506.99 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 1.875070472533741e-11
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8669907202789793
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 1.779800553664997e-05
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 67.98 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 3048.54 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 1.5553455513746386e-11
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8895271763106959
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 3.932787036614283e-06
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.55 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Prediction finished in 245.65 sec.
[2m[36m(pid=47)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=47)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                | node_ip_address   | model_params                                 |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------------------------------------|
|  0 |  1.64679e-11 |         0.887318 |  6.10273e-06 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/98441a856c22468e9eadef19b8f77335/train/766ae16238d4430a98d435d1850b5fd3/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_bertbaseuncased_lg'}       |
|  1 |  2.04349e-11 |         0.864339 |  1.63259e-05 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/c35d16fa51b44dcfa3c86fc2b7e53519/train/39d88d0af2de4c909469000972815ac6/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_xlnetbasecased_lg'}        |
|  2 |  1.87507e-11 |         0.866991 |  1.7798e-05  | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/c495f359d81646598b8227619bc6f4a3/train/118aa9c068c04e298bc3b90c62ae43dd/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_robertabase_lg'}           |
|  3 |  1.55535e-11 |         0.889527 |  3.93279e-06 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/e77c33a427ab4eb989c253c8c01bfba2/train/b2f825d6726d437a89fc325ac0d602d7/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_distilbertbaseuncased_lg'} |
```
Metrics:
--------
Weighted F1 Score: 0.8366399148510042
Weighted Precision Score: 0.8400197447681031
Weighted Recall Score: 0.8359001593202336
Accuracy: 0.8359001593202336

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.83      0.71      0.77       319
           comp.graphics       0.78      0.76      0.77       389
 comp.os.ms-windows.misc       0.79      0.74      0.77       394
comp.sys.ibm.pc.hardware       0.65      0.80      0.72       392
   comp.sys.mac.hardware       0.81      0.79      0.80       385
          comp.windows.x       0.90      0.83      0.86       395
            misc.forsale       0.89      0.88      0.88       390
               rec.autos       0.86      0.87      0.87       396
         rec.motorcycles       0.90      0.87      0.88       398
      rec.sport.baseball       0.93      0.92      0.93       397
        rec.sport.hockey       0.96      0.96      0.96       399
               sci.crypt       0.93      0.88      0.90       396
         sci.electronics       0.74      0.75      0.75       393
                 sci.med       0.91      0.92      0.91       396
               sci.space       0.88      0.89      0.88       394
  soc.religion.christian       0.89      0.93      0.91       398
      talk.politics.guns       0.70      0.85      0.77       364
   talk.politics.mideast       0.94      0.88      0.91       376
      talk.politics.misc       0.75      0.67      0.71       310
      talk.religion.misc       0.67      0.72      0.70       251

                accuracy                           0.84      7532
               macro avg       0.84      0.83      0.83      7532
            weighted avg       0.84      0.84      0.84      7532


```

![Results](spacy-transformers/plot.png)
---