# Results: DistilBERT
```
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/96f7bae68e734282822a4602f4c255d3'
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Build finished in 0.58 sec.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Training finished in 608.37 sec.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:  Validation loss: 0.01350625830246082
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.9010163499779055
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:  Training loss: 0.0018005909378853455
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5f3bf613cb13435dbb149fcf448565ab'
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Build finished in 0.26 sec.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Training finished in 551.27 sec.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:  Validation loss: 0.014563428090701732
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8846663720724701
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:  Training loss: 0.0032057813761290524
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/cc1d2b0a3105467b8dce53595d0d9d96'
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Build finished in 0.25 sec.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1937)[0m INFO:gobbli.experiment.base:Prediction finished in 46.88 sec.
[2m[36m(pid=1937)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1937)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                                                          |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------------------------------------|
|  0 |    0.0135063 |         0.901016 |   0.00180059 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/96f7bae68e734282822a4602f4c255d3/train/76c0eb0eeaee4820b0fe7365219fe8db/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased'}                 |
|  1 |    0.0145634 |         0.884666 |   0.00320578 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5f3bf613cb13435dbb149fcf448565ab/train/54b34a27a09f44228f5cffe2c7a48e29/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased-distilled-squad'} |
```
Metrics:
--------
Weighted F1 Score: 0.8268778917348241
Weighted Precision Score: 0.8314467456648094
Weighted Recall Score: 0.8248805098247477
Accuracy: 0.8248805098247477

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

![Results](DistilBERT/plot.png)
---