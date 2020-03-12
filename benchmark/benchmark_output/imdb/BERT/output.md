# Results: BERT
```
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Build finished in 0.37 sec.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:Training finished in 1702.40 sec.
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Validation loss: 0.55245966
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8882
[2m[36m(pid=45)[0m INFO:gobbli.experiment.base:  Training loss: 0.55339825
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Build finished in 0.59 sec.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Training finished in 1699.59 sec.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:  Validation loss: 0.63585734
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.874
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:  Training loss: 0.6350079
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Build finished in 0.38 sec.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Training finished in 1702.41 sec.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:  Validation loss: 0.5771891
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8856
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:  Training loss: 0.57802886
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Build finished in 0.84 sec.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=44)[0m INFO:gobbli.experiment.base:Prediction finished in 226.42 sec.
[2m[36m(pid=44)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=44)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                               | node_ip_address   | model_params                                               |
|---:|-------------:|-----------------:|-------------:|:---------------|:-------------------------------------------------------------------------------------------------------------------------|:------------------|:-----------------------------------------------------------|
|  0 |     0.55246  |           0.8882 |     0.553398 | ['neg', 'pos'] | benchmark_data/model/BERT/d552955c2bb24d9494adf5f67a5bc236/train/c04d9eaab55c4b289daf1fec0ad83a85/output/model.ckpt-3125 | 172.80.10.2       | {'bert_model': 'bert-base-uncased', 'max_seq_length': 128} |
|  1 |     0.635857 |           0.874  |     0.635008 | ['neg', 'pos'] | benchmark_data/model/BERT/f0404a440eeb464290d86e69f7104dd1/train/2acd79e6f58f499fa03fb07d89ea510b/output/model.ckpt-3125 | 172.80.10.2       | {'bert_model': 'bert-base-cased', 'max_seq_length': 128}   |
|  2 |     0.577189 |           0.8856 |     0.578029 | ['neg', 'pos'] | benchmark_data/model/BERT/cf6fd83548fe4a1580097f1325596dfa/train/76bdfaa9dd704934be341b499df1f4ce/output/model.ckpt-3125 | 172.80.10.2       | {'bert_model': 'scibert-uncased', 'max_seq_length': 128}   |
```
Metrics:
--------
Weighted F1 Score: 0.8795931909467853
Weighted Precision Score: 0.87968588555481
Weighted Recall Score: 0.8796
Accuracy: 0.8796

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.89      0.87      0.88     12500
         pos       0.87      0.89      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000


```

![Results](BERT/plot.png)
---