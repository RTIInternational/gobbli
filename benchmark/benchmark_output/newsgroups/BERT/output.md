# Results: BERT
```
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/9b476a087f9245d9a7aede442340941d'
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Build finished in 0.39 sec.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Training finished in 822.68 sec.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Validation loss: 0.34764564
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9085285
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Training loss: 0.34746882
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/ed48cea68ac44318ab7a3b83f72b3855'
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Training finished in 796.80 sec.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Validation loss: 0.45498648
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8802475
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Training loss: 0.4559938
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/7632a22eb93d45189dd67354e3688b74'
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Build finished in 0.24 sec.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Training finished in 800.09 sec.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Validation loss: 0.38081968
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.90499336
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:  Training loss: 0.38104215
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:BERT initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/504d21088f0e4ffd920a7b29950a3672'
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Build finished in 0.29 sec.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=516)[0m INFO:gobbli.experiment.base:Prediction finished in 74.98 sec.
[2m[36m(pid=516)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=516)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                   | node_ip_address   | model_params                                               |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-----------------------------------------------------------|
|  0 |     0.347646 |         0.908528 |     0.347469 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/9b476a087f9245d9a7aede442340941d/train/1974eed027bf41cca2b443dd39329f48/output/model.ckpt-1414 | 172.80.10.2       | {'bert_model': 'bert-base-uncased', 'max_seq_length': 128} |
|  1 |     0.454986 |         0.880247 |     0.455994 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/ed48cea68ac44318ab7a3b83f72b3855/train/b426a39280674d0a8c80e4f5d8d38ccc/output/model.ckpt-1414 | 172.80.10.2       | {'bert_model': 'bert-base-cased', 'max_seq_length': 128}   |
|  2 |     0.38082  |         0.904993 |     0.381042 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/BERT/7632a22eb93d45189dd67354e3688b74/train/84ff95a81ead4e51bcd3796b95492186/output/model.ckpt-1414 | 172.80.10.2       | {'bert_model': 'scibert-uncased', 'max_seq_length': 128}   |
```
Metrics:
--------
Weighted F1 Score: 0.837437045888832
Weighted Precision Score: 0.8406616781349776
Weighted Recall Score: 0.8370950610727562
Accuracy: 0.8370950610727562

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

![Results](BERT/plot.png)
---