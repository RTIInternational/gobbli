# Results: MTDNN
```
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Build finished in 0.61 sec.
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=217)[0m /code/gobbli/model/mtdnn/model.py:193: UserWarning: MT-DNN model does not support separate validation batch size; using train batch size '16' for both training and validation.
[2m[36m(pid=217)[0m   "MT-DNN model does not support separate validation batch size; "
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Training finished in 1261.09 sec.
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:  Validation loss: 2.2728805541992188
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8674326115775518
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:  Training loss: 2.1877872943878174
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Build finished in 1.00 sec.
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=217)[0m INFO:gobbli.experiment.base:Prediction finished in 68.12 sec.
[2m[36m(pid=217)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=217)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                           | node_ip_address   | model_params                                          |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------|
|  0 |      2.27288 |         0.867433 |      2.18779 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/MTDNN/64f7210b2c1a41fb9ba896ecf29fabdb/train/a7bbd062f8984bbd857f2d84dc37f0b1/output/model_4.pt | 172.80.10.2       | {'max_seq_length': 128, 'mtdnn_model': 'mt-dnn-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8035832681908751
Weighted Precision Score: 0.806473034095344
Weighted Recall Score: 0.8033722782793414
Accuracy: 0.8033722782793414

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.72      0.63      0.67       319
           comp.graphics       0.76      0.77      0.76       389
 comp.os.ms-windows.misc       0.75      0.78      0.76       394
comp.sys.ibm.pc.hardware       0.59      0.67      0.63       392
   comp.sys.mac.hardware       0.82      0.74      0.77       385
          comp.windows.x       0.85      0.80      0.82       395
            misc.forsale       0.90      0.89      0.89       390
               rec.autos       0.86      0.86      0.86       396
         rec.motorcycles       0.88      0.74      0.80       398
      rec.sport.baseball       0.87      0.87      0.87       397
        rec.sport.hockey       0.93      0.91      0.92       399
               sci.crypt       0.91      0.86      0.88       396
         sci.electronics       0.71      0.81      0.76       393
                 sci.med       0.90      0.94      0.92       396
               sci.space       0.90      0.91      0.90       394
  soc.religion.christian       0.85      0.92      0.88       398
      talk.politics.guns       0.65      0.77      0.71       364
   talk.politics.mideast       0.91      0.88      0.90       376
      talk.politics.misc       0.61      0.57      0.59       310
      talk.religion.misc       0.64      0.59      0.61       251

                accuracy                           0.80      7532
               macro avg       0.80      0.80      0.80      7532
            weighted avg       0.81      0.80      0.80      7532


```

![Results](MTDNN/plot.png)
---