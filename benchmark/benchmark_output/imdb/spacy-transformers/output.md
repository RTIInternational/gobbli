# Results: spacy-transformers
```
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Build finished in 0.36 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=665)[0m /code/gobbli/model/spacy/model.py:174: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=665)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Training finished in 7872.90 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation loss: 4.758294025464238e-12
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.3631999999992736
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Training loss: 3.0240254466275473e-05
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Build finished in 0.34 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Training finished in 8524.58 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation loss: 2.855722591732501e-12
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7397999999985204
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Training loss: 8.88249805138912e-05
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Build finished in 0.28 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Training finished in 7744.93 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation loss: 4.001228504080245e-12
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.5013999999989972
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Training loss: 0.00012297899288823828
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Build finished in 0.39 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Training finished in 6012.45 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation loss: 1.9479770685393303e-12
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8503999999982992
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:  Training loss: 4.971897644175955e-06
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Build finished in 0.29 sec.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=665)[0m INFO:gobbli.experiment.base:Prediction finished in 705.86 sec.
[2m[36m(pid=665)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=665)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                | node_ip_address   | model_params                                 |
|---:|-------------:|-----------------:|-------------:|:---------------|:--------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------------------------------------|
|  0 |  4.75829e-12 |           0.3632 |  3.02403e-05 | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/79329ba7167b44d894ebaff98f09e937/train/fa7e77298b1c4b2b840bf3a68c084eec/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_bertbaseuncased_lg'}       |
|  1 |  2.85572e-12 |           0.7398 |  8.8825e-05  | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/b9d9c344d270433fb1f6415c78fe41cc/train/b9258f302b644c78a6357da377a7b191/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_xlnetbasecased_lg'}        |
|  2 |  4.00123e-12 |           0.5014 |  0.000122979 | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/39853179da05474e82d29749f585c031/train/f3e660cad88e40b1ae9077ae1714dacd/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_robertabase_lg'}           |
|  3 |  1.94798e-12 |           0.8504 |  4.9719e-06  | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/9a8ea90c6298424f8d2c94cc64a6e22e/train/6c1b6fe0d8684ca38c2537c59b361b81/output/checkpoint | 172.80.10.2       | {'model': 'en_trf_distilbertbaseuncased_lg'} |
```
Metrics:
--------
Weighted F1 Score: 0.8372719349367306
Weighted Precision Score: 0.8373468778040759
Weighted Recall Score: 0.83728
Accuracy: 0.83728

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.83      0.84      0.84     12500
         pos       0.84      0.83      0.84     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000


```

![Results](spacy-transformers/plot.png)
---