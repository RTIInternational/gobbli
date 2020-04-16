# Results: spacy-transformers
```
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/9ffe6222598f4e8e884fbe60ac458d0e'
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Build finished in 0.28 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3086)[0m /code/gobbli/model/spacy/model.py:176: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=3086)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Training finished in 4155.86 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation loss: 1.516424105154493e-11
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9897702165265158
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Training loss: 5.762817107745332e-06
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/52f14545ff5c4e159dda1568908a66d0'
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Build finished in 0.40 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Training finished in 4824.18 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation loss: 1.5262768806949932e-11
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9900353513033606
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Training loss: 1.6731242336570725e-05
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/ffc52be82d484a08a2a726314b9b5018'
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Build finished in 0.33 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Training finished in 4480.74 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation loss: 1.731408810206097e-11
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9877817057001795
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Training loss: 2.0574636453659296e-05
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/21d45d7e540b429ab452709132af359d'
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Build finished in 0.31 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Training finished in 2952.86 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation loss: 1.484312000653494e-11
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9898585947854641
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:  Training loss: 3.2828676847331087e-06
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:SpaCyModel initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/b4e623d017d74af6a5c2c24b48b8a378'
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Build finished in 0.30 sec.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=3086)[0m INFO:gobbli.experiment.base:Prediction finished in 250.64 sec.
[2m[36m(pid=3086)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=3086)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                    | node_ip_address   | model_params                                 |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------------------------------------|
|  0 |  1.51642e-11 |         0.98977  |  5.76282e-06 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/9ffe6222598f4e8e884fbe60ac458d0e/train/dc13a2caab3244b7a0bd96e3a473b82a/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_bertbaseuncased_lg'}       |
|  1 |  1.52628e-11 |         0.990035 |  1.67312e-05 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/52f14545ff5c4e159dda1568908a66d0/train/e9dedfc5333940de99056c2d802d78aa/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_xlnetbasecased_lg'}        |
|  2 |  1.73141e-11 |         0.987782 |  2.05746e-05 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/ffc52be82d484a08a2a726314b9b5018/train/a12ec20dde5e404881a0fec4abc36216/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_robertabase_lg'}           |
|  3 |  1.48431e-11 |         0.989859 |  3.28287e-06 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/SpaCyModel/21d45d7e540b429ab452709132af359d/train/2757850db67145d2afabe2161bce9905/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_distilbertbaseuncased_lg'} |
```
Metrics:
--------
Weighted F1 Score: 0.8232098408476677
Weighted Precision Score: 0.8253413271393194
Weighted Recall Score: 0.8228890069038768
Accuracy: 0.8228890069038768

Classification Report:
----------------------
                          precision    recall  f1-score   support

             alt.atheism       0.00      0.00      0.00         0
           comp.graphics       0.00      0.00      0.00         0
 comp.os.ms-windows.misc       0.00      0.00      0.00         0
comp.sys.ibm.pc.hardware       0.00      0.00      0.00         0
   comp.sys.mac.hardware       0.00      0.00      0.00         0
          comp.windows.x       0.00      0.00      0.00         0
            misc.forsale       0.00      0.00      0.00         0
               rec.autos       0.00      0.00      0.00         0
         rec.motorcycles       0.00      0.00      0.00         0
      rec.sport.baseball       0.00      0.00      0.00         0
        rec.sport.hockey       0.00      0.00      0.00         0
               sci.crypt       0.00      0.00      0.00         0
         sci.electronics       0.00      0.00      0.00         0
                 sci.med       0.00      0.00      0.00         0
               sci.space       0.00      0.00      0.00         0
  soc.religion.christian       0.00      0.00      0.00         0
      talk.politics.guns       0.00      0.00      0.00         0
   talk.politics.mideast       0.00      0.00      0.00         0
      talk.politics.misc       0.00      0.00      0.00         0
      talk.religion.misc       0.00      0.00      0.00         0

               micro avg       0.00      0.00      0.00         0
               macro avg       0.00      0.00      0.00         0
            weighted avg       0.00      0.00      0.00         0


```

![Results](spacy-transformers/plot.png)
---