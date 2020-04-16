# Results: XLM
```
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/d02f90c349fd428dabbeefee64cbb0cc'
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Build finished in 0.26 sec.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Training finished in 1379.16 sec.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:  Validation loss: 0.09383381060989252
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.048608042421564294
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:  Training loss: 0.20049872220306683
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/3210c170deda45e4b454c58bfe07b819'
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Build finished in 0.20 sec.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Training finished in 819.81 sec.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:  Validation loss: 0.023119816845476336
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8532920901458241
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:  Training loss: 0.006614376033375956
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/f73e9bfca4414c04bfe29e5532f21459'
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Build finished in 0.20 sec.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1247)[0m INFO:gobbli.experiment.base:Prediction finished in 55.19 sec.
[2m[36m(pid=1247)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1247)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0938338 |         0.048608 |   0.200499   | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/d02f90c349fd428dabbeefee64cbb0cc/train/708b725b38a24ae6afc5e2fee3cf8764/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-mlm-tlm-xnli15-1024'} |
|  1 |    0.0231198 |         0.853292 |   0.00661438 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/3210c170deda45e4b454c58bfe07b819/train/e6588a51b3f4499397908ff8cb687f67/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-clm-ende-1024'}       |
```
Metrics:
--------
Weighted F1 Score: 0.7663560244706216
Weighted Precision Score: 0.7696143964835898
Weighted Recall Score: 0.7656664896441848
Accuracy: 0.7656664896441848

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

![Results](XLM/plot.png)
---