# Results: RoBERTa
```
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/384a943be35e45108d5c085a64a53755'
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Build finished in 0.46 sec.
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Training finished in 916.25 sec.
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:  Validation loss: 0.016980514437342133
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.895271763146266
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:  Training loss: 0.005296037527501547
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/3de0f42cb83f40c691407612654ebf1c'
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Build finished in 0.25 sec.
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1718)[0m INFO:gobbli.experiment.base:Prediction finished in 51.55 sec.
[2m[36m(pid=1718)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1718)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                    | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                            |
|---:|-------------:|-----------------:|-------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------|
|  0 |    0.0169805 |         0.895272 |   0.00529604 | False        | ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/384a943be35e45108d5c085a64a53755/train/aa4d2d58345e4ac7b3740e076291407b/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Roberta', 'transformer_weights': 'roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8150377440478993
Weighted Precision Score: 0.8226010055432904
Weighted Recall Score: 0.8134625597450876
Accuracy: 0.8134625597450876

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

![Results](RoBERTa/plot.png)
---