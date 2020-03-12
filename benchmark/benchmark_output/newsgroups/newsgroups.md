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
# Results: FastText
```
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Build finished in 1.26 sec.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:Build finished in 1.25 sec.
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Build finished in 1.26 sec.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Build finished in 1.25 sec.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Training finished in 315.01 sec.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8497569597878922
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8497569597878922
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:  Training loss: 1.57841
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Build finished in 0.65 sec.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:Training finished in 317.94 sec.
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:  Validation loss: -0.41802916482545294
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.41802916482545294
[2m[36m(pid=378)[0m INFO:gobbli.experiment.base:  Training loss: 2.489756
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Build finished in 0.34 sec.
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Training finished in 318.80 sec.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:  Validation loss: -0.7255855059655325
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7255855059655325
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:  Training loss: 1.632553
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Build finished in 0.38 sec.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Training finished in 327.84 sec.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:  Validation loss: -0.7901016349977905
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7901016349977905
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:  Training loss: 1.106578
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Build finished in 0.65 sec.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:Training finished in 341.45 sec.
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8453380468404772
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8453380468404772
[2m[36m(pid=379)[0m INFO:gobbli.experiment.base:  Training loss: 1.178128
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:Training finished in 345.06 sec.
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:  Validation loss: -0.7061422889969068
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7061422889969068
[2m[36m(pid=380)[0m INFO:gobbli.experiment.base:  Training loss: 1.903518
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:Training finished in 337.78 sec.
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:  Validation loss: -0.7578435704816615
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7578435704816615
[2m[36m(pid=377)[0m INFO:gobbli.experiment.base:  Training loss: 1.544925
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Training finished in 349.15 sec.
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:  Validation loss: -0.34953601414052143
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.34953601414052143
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:  Training loss: 2.474048
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Build finished in 0.35 sec.
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=476)[0m INFO:gobbli.experiment.base:Prediction finished in 3.37 sec.
[2m[36m(pid=476)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=476)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                         | node_ip_address   | model_params                              |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------|
|  0 |    -0.725586 |         0.725586 |      1.63255 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/ecf43233518d44049671621b232e27d9/train/d674f3ea30a64a659ecf38222d28623f/output/model | 172.80.10.2       | {'dim': 100, 'lr': 0.5, 'word_ngrams': 1} |
|  1 |    -0.418029 |         0.418029 |      2.48976 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/945e185dcf9e495ea197301f7377a443/train/5114c2313d9a4e13bfa367ab869c1aca/output/model | 172.80.10.2       | {'dim': 100, 'lr': 0.5, 'word_ngrams': 2} |
|  2 |    -0.849757 |         0.849757 |      1.57841 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/1cf1cbc011bb48d092fe349d8d27afdb/train/a84dac18845842b3aa28fff084ee0c52/output/model | 172.80.10.2       | {'dim': 100, 'lr': 1.0, 'word_ngrams': 1} |
|  3 |    -0.790102 |         0.790102 |      1.10658 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/76ccb6afecc84b79a70b4c5c801fc121/train/bb91e6e7ea2e48e2995cb5672b4205a7/output/model | 172.80.10.2       | {'dim': 100, 'lr': 1.0, 'word_ngrams': 2} |
|  4 |    -0.706142 |         0.706142 |      1.90352 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/10997ef754e34abda7764ff849f823e0/train/2d437e173596465e998b5bfc7c87dfb7/output/model | 172.80.10.2       | {'dim': 300, 'lr': 0.5, 'word_ngrams': 1} |
|  5 |    -0.349536 |         0.349536 |      2.47405 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/f833f73cfd9e40ba8735fc30258c39f8/train/3909ff17d2af4414a88a0349407ca1aa/output/model | 172.80.10.2       | {'dim': 300, 'lr': 0.5, 'word_ngrams': 2} |
|  6 |    -0.845338 |         0.845338 |      1.17813 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/745d9f3117824dd3bc987a3c6999ad6a/train/6901cf42586f4911bb6661aa42f71ad2/output/model | 172.80.10.2       | {'dim': 300, 'lr': 1.0, 'word_ngrams': 1} |
|  7 |    -0.757844 |         0.757844 |      1.54493 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/FastText/448abc9fd8184c2c9f9789af4e09f281/train/165d24429e8748249745f3b7aa56127c/output/model | 172.80.10.2       | {'dim': 300, 'lr': 1.0, 'word_ngrams': 2} |
```
Metrics:
--------
Weighted F1 Score: 0.746025332814778
Weighted Precision Score: 0.7510731273540894
Weighted Recall Score: 0.7441582580987786
Accuracy: 0.7441582580987786

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.59      0.65      0.62       319
           comp.graphics       0.65      0.66      0.65       389
 comp.os.ms-windows.misc       0.72      0.64      0.68       394
comp.sys.ibm.pc.hardware       0.60      0.66      0.63       392
   comp.sys.mac.hardware       0.74      0.70      0.72       385
          comp.windows.x       0.81      0.70      0.75       395
            misc.forsale       0.81      0.83      0.82       390
               rec.autos       0.78      0.83      0.80       396
         rec.motorcycles       0.87      0.86      0.87       398
      rec.sport.baseball       0.85      0.85      0.85       397
        rec.sport.hockey       0.94      0.88      0.91       399
               sci.crypt       0.87      0.83      0.85       396
         sci.electronics       0.61      0.68      0.64       393
                 sci.med       0.78      0.76      0.77       396
               sci.space       0.86      0.84      0.85       394
  soc.religion.christian       0.79      0.79      0.79       398
      talk.politics.guns       0.68      0.79      0.73       364
   talk.politics.mideast       0.93      0.76      0.84       376
      talk.politics.misc       0.49      0.58      0.53       310
      talk.religion.misc       0.44      0.39      0.41       251

                accuracy                           0.74      7532
               macro avg       0.74      0.74      0.74      7532
            weighted avg       0.75      0.74      0.75      7532


```

![Results](FastText/plot.png)
---
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
# Results: spaCy
```
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Build finished in 11.60 sec.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=45)[0m /code/gobbli/model/spacy/model.py:174: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=45)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Training finished in 450.61 sec.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Validation loss: 1.9342751979099198e-11
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8608042421526257
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Training loss: 2.8173636781991426e-06
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Build finished in 203.45 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=46)[0m /code/gobbli/model/spacy/model.py:174: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=46)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Training finished in 353.72 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation loss: 2.0184255308041736e-11
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8555015466157513
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Training loss: 3.493727465702871e-06
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Build finished in 0.63 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Training finished in 590.44 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation loss: 3.361362604766144e-11
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7118868758254004
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Training loss: 1.0983899320602732e-05
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Build finished in 0.42 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Training finished in 475.64 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation loss: 2.4078957318736282e-11
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7976137870048714
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Training loss: 7.640532423431264e-06
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Build finished in 0.48 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Training finished in 625.89 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation loss: 2.6194864414757412e-11
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.782589482983727
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Training loss: 9.920230171940159e-06
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Build finished in 0.30 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Training finished in 489.78 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation loss: 2.633931998221425e-11
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7852408307521642
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:  Training loss: 1.0416920807455892e-05
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Build finished in 0.82 sec.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=46)[0m INFO:gobbli.experiment.base:Prediction finished in 50.53 sec.
[2m[36m(pid=46)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=46)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                | node_ip_address   | model_params                                              |
|---:|-------------:|-----------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|:------------------|:----------------------------------------------------------|
|  0 |  1.93428e-11 |         0.860804 |  2.81736e-06 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/100f9daf5f0e4234a787140162b1fb7f/train/39273ece4ff940cd9870ee77ea6d58ed/output/checkpoint | 172.80.10.2       | {'architecture': 'bow', 'model': 'en_core_web_sm'}        |
|  1 |  2.01843e-11 |         0.855502 |  3.49373e-06 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/95125315ca254f19ba59864eb6a0516a/train/1e4366ac1d59482a82b7930af5f6d9dd/output/checkpoint | 172.80.10.2       | {'architecture': 'bow', 'model': 'en_core_web_lg'}        |
|  2 |  3.36136e-11 |         0.711887 |  1.09839e-05 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/aa5a5e73cec845a6adcb36bc69dd2c60/train/ade49e105b374be684ff47e734e6f9c4/output/checkpoint | 172.80.10.2       | {'architecture': 'simple_cnn', 'model': 'en_core_web_sm'} |
|  3 |  2.4079e-11  |         0.797614 |  7.64053e-06 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/7fca7c7ac2ac4ba597b95557f0281f5c/train/adb8170ab4ff4674a4edf1c80805943f/output/checkpoint | 172.80.10.2       | {'architecture': 'simple_cnn', 'model': 'en_core_web_lg'} |
|  4 |  2.61949e-11 |         0.782589 |  9.92023e-06 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/ae5cad1f9e444b00b70b6451dad209df/train/e2419b5dc9704842981beb789c42baa3/output/checkpoint | 172.80.10.2       | {'architecture': 'ensemble', 'model': 'en_core_web_sm'}   |
|  5 |  2.63393e-11 |         0.785241 |  1.04169e-05 | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | benchmark_data/model/SpaCyModel/05cf21870b8e42dd805f01ed233fde48/train/03fb74a643a8460aa0bc7b092ee6686a/output/checkpoint | 172.80.10.2       | {'architecture': 'ensemble', 'model': 'en_core_web_lg'}   |
```
Metrics:
--------
Weighted F1 Score: 0.740745356478981
Weighted Precision Score: 0.7423985753517668
Weighted Recall Score: 0.7413701540095592
Accuracy: 0.7413701540095592

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.68      0.66      0.67       319
           comp.graphics       0.54      0.63      0.58       389
 comp.os.ms-windows.misc       0.62      0.63      0.62       394
comp.sys.ibm.pc.hardware       0.63      0.64      0.63       392
   comp.sys.mac.hardware       0.69      0.69      0.69       385
          comp.windows.x       0.79      0.69      0.74       395
            misc.forsale       0.76      0.85      0.80       390
               rec.autos       0.80      0.79      0.79       396
         rec.motorcycles       0.80      0.87      0.83       398
      rec.sport.baseball       0.84      0.84      0.84       397
        rec.sport.hockey       0.89      0.90      0.89       399
               sci.crypt       0.82      0.79      0.81       396
         sci.electronics       0.66      0.62      0.64       393
                 sci.med       0.76      0.70      0.73       396
               sci.space       0.84      0.85      0.85       394
  soc.religion.christian       0.80      0.83      0.81       398
      talk.politics.guns       0.72      0.79      0.76       364
   talk.politics.mideast       0.89      0.88      0.89       376
      talk.politics.misc       0.69      0.55      0.61       310
      talk.religion.misc       0.54      0.50      0.52       251

                accuracy                           0.74      7532
               macro avg       0.74      0.73      0.73      7532
            weighted avg       0.74      0.74      0.74      7532


```

![Results](spaCy/plot.png)
---
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