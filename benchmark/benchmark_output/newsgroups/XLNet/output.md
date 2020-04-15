# Results: XLNet
```
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5a8933659ab5448a8aa2067cd48a6c96'
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Build finished in 0.21 sec.
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Training finished in 1095.99 sec.
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:  Validation loss: 0.02455309041295492
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7494476358815731
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:  Training loss: 0.04835376695478414
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/849f8310560f469b992ced757abf0ddf'
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Build finished in 0.23 sec.
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1477)[0m INFO:gobbli.experiment.base:Prediction finished in 68.84 sec.
[2m[36m(pid=1477)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1477)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                              |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:--------------------------------------------------------------------------|
|  0 |    0.0245531 |         0.749448 |    0.0483538 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/5a8933659ab5448a8aa2067cd48a6c96/train/d5e726b926414724a71bdc398f410b96/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLNet', 'transformer_weights': 'xlnet-base-cased'} |
```
Metrics:
--------
Weighted F1 Score: 0.6676947776035
Weighted Precision Score: 0.669441323343694
Weighted Recall Score: 0.6775092936802974
Accuracy: 0.6775092936802974

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

![Results](XLNet/plot.png)
---