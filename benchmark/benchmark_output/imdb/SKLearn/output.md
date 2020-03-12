# Results: SKLearn
```
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Training finished in 14.84 sec.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8863778586526605
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8864
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Training loss: -0.9309477856498368
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Build finished in 0.00 sec.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Prediction finished in 6.54 sec.
[2m[36m(pid=354)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=354)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                             | node_ip_address   | model_params   |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------|:------------------|:---------------|
|  0 |    -0.886378 |           0.8864 |    -0.930948 | ['neg', 'pos'] | benchmark_data/model/SKLearnClassifier/88785fc4fc37491e934045d37cf08752/train/ebe02fe975dc428b93d707744eda023e/output/estimator.joblib | 172.80.10.2       | {}             |
```
Metrics:
--------
Weighted F1 Score: 0.87775924817048
Weighted Precision Score: 0.8777692937290567
Weighted Recall Score: 0.87776
Accuracy: 0.87776

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.88      0.88      0.88     12500
         pos       0.88      0.88      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000


```

![Results](SKLearn/plot.png)
---