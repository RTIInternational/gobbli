# Results: XLM-RoBERTa
```
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/a5320a7da0e44f0498864c0110daa78e'
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Build finished in 0.21 sec.
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Training finished in 799.86 sec.
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:  Validation loss: 0.018771470712445976
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8329650905877154
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:  Training loss: 0.013683362611551045
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/131363db56e84d7194e7c5e0e04e3966'
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Build finished in 0.19 sec.
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=2398)[0m INFO:gobbli.experiment.base:Prediction finished in 51.98 sec.
[2m[36m(pid=2398)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=2398)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0187715 |         0.832965 |    0.0136834 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/a5320a7da0e44f0498864c0110daa78e/train/78d3f1a2616e4adcb474c10fd66d7337/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLMRoberta', 'transformer_weights': 'xlm-roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.7537213915386661
Weighted Precision Score: 0.7563358589480229
Weighted Recall Score: 0.7529208709506108
Accuracy: 0.7529208709506108

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

![Results](XLM-RoBERTa/plot.png)
---