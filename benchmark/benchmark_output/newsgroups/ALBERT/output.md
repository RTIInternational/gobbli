# Results: ALBERT
```
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/4751263bef4044b888874318bf8f8406'
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Training finished in 658.77 sec.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:  Validation loss: 0.016683096116325576
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8448961555457357
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:  Training loss: 0.00854644861593785
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/03e72ec7e47c444ea0c3b004b794078e'
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Build finished in 0.19 sec.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Training finished in 723.89 sec.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:  Validation loss: 0.03625400944510446
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.6703490941228458
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:  Training loss: 0.031885856687001876
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/ef11400c59324aa3a37e043d39e36741'
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Build finished in 0.25 sec.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=2164)[0m INFO:gobbli.experiment.base:Prediction finished in 44.44 sec.
[2m[36m(pid=2164)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=2164)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                             |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------|
|  0 |    0.0166831 |         0.844896 |   0.00854645 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/4751263bef4044b888874318bf8f8406/train/019a5e6ce7e94ba7a0da083d7ab6e7ac/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v1'} |
|  1 |    0.036254  |         0.670349 |   0.0318859  | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/03e72ec7e47c444ea0c3b004b794078e/train/1c506e63d1b3479b87f5bb2c405d9772/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v2'} |
```
Metrics:
--------
Weighted F1 Score: 0.7809736147217773
Weighted Precision Score: 0.7840864983706365
Weighted Recall Score: 0.780801911842804
Accuracy: 0.780801911842804

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

![Results](ALBERT/plot.png)
---