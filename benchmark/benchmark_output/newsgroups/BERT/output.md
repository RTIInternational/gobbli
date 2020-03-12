# Results: BERT
```
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Build finished in 0.44 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Training finished in 829.07 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation loss: 0.37740374
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9089704
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Training loss: 0.37608728
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.47 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 819.08 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.76031363
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7909854
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.7595904
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.62 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 813.59 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.37356544
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9036677
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.37227306
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.50 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Prediction finished in 87.91 sec.
[2m[36m(pid=47)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=47)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                               | node_ip_address   | model_params                                               |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|:------------------|:-----------------------------------------------------------|
|  0 |     0.377404 |         0.90897  |     0.376087 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/BERT/ddb393ea961049e6b87173edced88f1d/train/6085548d1a124f9f9dedfd8d9237d9c4/output/model.ckpt-1414 | 172.80.10.2       | {'bert_model': 'bert-base-uncased', 'max_seq_length': 128} |
|  1 |     0.760314 |         0.790985 |     0.75959  | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/BERT/2e7a2ec853f34eafb46c9803510c2187/train/8dbc0a913cae4c0e997c6e2a462108a5/output/model.ckpt-1414 | 172.80.10.2       | {'bert_model': 'bert-base-cased', 'max_seq_length': 128}   |
|  2 |     0.373565 |         0.903668 |     0.372273 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/BERT/671312b30b1e44b5ab7e4f794535947f/train/d8ba12ebc04944719d28e2571d225d52/output/model.ckpt-1414 | 172.80.10.2       | {'bert_model': 'scibert-uncased', 'max_seq_length': 128}   |
```
Metrics:
--------
Weighted F1 Score: 0.8343736377594676
Weighted Precision Score: 0.8385946437559766
Weighted Recall Score: 0.8340414232607541
Accuracy: 0.8340414232607541

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.76      0.66      0.71       319
           comp.graphics       0.79      0.78      0.78       389
 comp.os.ms-windows.misc       0.78      0.81      0.80       394
comp.sys.ibm.pc.hardware       0.64      0.71      0.67       392
   comp.sys.mac.hardware       0.80      0.77      0.79       385
          comp.windows.x       0.89      0.83      0.86       395
            misc.forsale       0.88      0.87      0.88       390
               rec.autos       0.90      0.90      0.90       396
         rec.motorcycles       0.88      0.84      0.86       398
      rec.sport.baseball       0.92      0.89      0.90       397
        rec.sport.hockey       0.96      0.93      0.95       399
               sci.crypt       0.90      0.93      0.91       396
         sci.electronics       0.78      0.82      0.80       393
                 sci.med       0.94      0.95      0.94       396
               sci.space       0.90      0.92      0.91       394
  soc.religion.christian       0.93      0.92      0.93       398
      talk.politics.guns       0.70      0.86      0.77       364
   talk.politics.mideast       0.94      0.89      0.91       376
      talk.politics.misc       0.78      0.55      0.64       310
      talk.religion.misc       0.56      0.71      0.63       251

                accuracy                           0.83      7532
               macro avg       0.83      0.83      0.83      7532
            weighted avg       0.84      0.83      0.83      7532


```

![Results](BERT/plot.png)
---