# Results: XLM
```
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Build finished in 0.47 sec.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Training finished in 3089.48 sec.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:  Validation loss: 0.02176531046628952
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.4906
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:  Training loss: 0.043933403673768044
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Build finished in 0.27 sec.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Training finished in 1796.83 sec.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:  Validation loss: 0.024198950758203865
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.828
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:  Training loss: 0.009087214053142816
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Build finished in 0.37 sec.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=551)[0m INFO:gobbli.experiment.base:Prediction finished in 233.77 sec.
[2m[36m(pid=551)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=551)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0217653 |           0.4906 |   0.0439334  | ['neg', 'pos'] | benchmark_data/model/Transformer/a2e0a156e390456b862cb737a1e84bc1/train/48476a321ccf4ea2875bbf60a82c7b32/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-mlm-tlm-xnli15-1024'} |
|  1 |    0.024199  |           0.828  |   0.00908721 | ['neg', 'pos'] | benchmark_data/model/Transformer/bf3df4408a6f4c93adc1388cd109264d/train/830bbf21e5284c959c4d88b1c95a56bd/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLM', 'transformer_weights': 'xlm-clm-ende-1024'}       |
```
Metrics:
--------
Weighted F1 Score: 0.33331549252537007
Weighted Precision Score: 0.3213785574246503
Weighted Recall Score: 0.4998
Accuracy: 0.4998

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.14      0.00      0.00     12500
         pos       0.50      1.00      0.67     12500

    accuracy                           0.50     25000
   macro avg       0.32      0.50      0.33     25000
weighted avg       0.32      0.50      0.33     25000


```

![Results](XLM/plot.png)
---