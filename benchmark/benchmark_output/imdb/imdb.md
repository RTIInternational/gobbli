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
# Results: MTDNN
```
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Build finished in 0.55 sec.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=207)[0m /code/gobbli/model/mtdnn/model.py:193: UserWarning: MT-DNN model does not support separate validation batch size; using train batch size '16' for both training and validation.
[2m[36m(pid=207)[0m   "MT-DNN model does not support separate validation batch size; "
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Training finished in 2652.60 sec.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:  Validation loss: 0.42352402210235596
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8866
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:  Training loss: 0.3258255124092102
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Build finished in 1.00 sec.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=207)[0m INFO:gobbli.experiment.base:Prediction finished in 189.84 sec.
[2m[36m(pid=207)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=207)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                           | node_ip_address   | model_params                                          |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------|
|  0 |     0.423524 |           0.8866 |     0.325826 | ['neg', 'pos'] | benchmark_data/model/MTDNN/22da8b980a894693b8812dc521417784/train/d81d2cdf873a4b23a8f7109a2fcd2df8/output/model_4.pt | 172.80.10.2       | {'max_seq_length': 128, 'mtdnn_model': 'mt-dnn-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8820389383504452
Weighted Precision Score: 0.8820537539351417
Weighted Recall Score: 0.88204
Accuracy: 0.88204

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.88      0.88      0.88     12500
         pos       0.88      0.89      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000


```

![Results](MTDNN/plot.png)
---
# Results: FastText
```
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Build finished in 0.70 sec.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Build finished in 0.70 sec.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Build finished in 0.63 sec.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Build finished in 0.75 sec.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Training finished in 322.28 sec.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8982
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8982
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:  Training loss: 0.258553
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Build finished in 0.36 sec.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Training finished in 327.91 sec.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8992
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8992
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:  Training loss: 0.260865
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Build finished in 0.26 sec.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Training finished in 329.68 sec.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8876
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8876
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Training loss: 0.300632
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Training finished in 329.81 sec.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8878
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8878
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:  Training loss: 0.292702
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Build finished in 0.44 sec.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Build finished in 0.37 sec.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:Training finished in 325.45 sec.
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8882
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8882
[2m[36m(pid=357)[0m INFO:gobbli.experiment.base:  Training loss: 0.333738
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:Training finished in 376.47 sec.
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8916
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8916
[2m[36m(pid=354)[0m INFO:gobbli.experiment.base:  Training loss: 0.319099
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:Training finished in 414.33 sec.
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8974
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8974
[2m[36m(pid=356)[0m INFO:gobbli.experiment.base:  Training loss: 0.218359
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Training finished in 417.83 sec.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:  Validation loss: -0.8946
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8946
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:  Training loss: 0.273403
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Build finished in 0.53 sec.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=355)[0m INFO:gobbli.experiment.base:Prediction finished in 9.24 sec.
[2m[36m(pid=355)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=355)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                         | node_ip_address   | model_params                              |
|---:|-------------:|-----------------:|-------------:|:---------------|:-------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------|
|  0 |      -0.8876 |           0.8876 |     0.300632 | ['neg', 'pos'] | benchmark_data/model/FastText/2bd8d69da37c4029b3535ddf32edbc1c/train/f94206879eb74907b139ab23fedcd26c/output/model | 172.80.10.2       | {'dim': 100, 'lr': 0.5, 'word_ngrams': 1} |
|  1 |      -0.8982 |           0.8982 |     0.258553 | ['neg', 'pos'] | benchmark_data/model/FastText/2a3a9af96efb4c47baa4c48f23253623/train/c96f489df90d4cdea9f5d1bf9bd5a856/output/model | 172.80.10.2       | {'dim': 100, 'lr': 0.5, 'word_ngrams': 2} |
|  2 |      -0.8878 |           0.8878 |     0.292702 | ['neg', 'pos'] | benchmark_data/model/FastText/83e00259e9b6464ab5c19fafff50c7a3/train/6c400b6e35e542aba24e63fbaf186bc2/output/model | 172.80.10.2       | {'dim': 100, 'lr': 1.0, 'word_ngrams': 1} |
|  3 |      -0.8992 |           0.8992 |     0.260865 | ['neg', 'pos'] | benchmark_data/model/FastText/b95b50e9cbf246dc800257fbe44d24ae/train/af6c14c956ed4bcc92f2ae00e3bee234/output/model | 172.80.10.2       | {'dim': 100, 'lr': 1.0, 'word_ngrams': 2} |
|  4 |      -0.8882 |           0.8882 |     0.333738 | ['neg', 'pos'] | benchmark_data/model/FastText/19c14eecf6a14edea9234c9f0f15c759/train/d83b2903ca22478dbb6ae0de6b5a7b5e/output/model | 172.80.10.2       | {'dim': 300, 'lr': 0.5, 'word_ngrams': 1} |
|  5 |      -0.8946 |           0.8946 |     0.273403 | ['neg', 'pos'] | benchmark_data/model/FastText/8bf691dc39ac466c8b0e540710146566/train/8261fd62582c4fa694140db3350e018e/output/model | 172.80.10.2       | {'dim': 300, 'lr': 0.5, 'word_ngrams': 2} |
|  6 |      -0.8916 |           0.8916 |     0.319099 | ['neg', 'pos'] | benchmark_data/model/FastText/012505f753b44a38920ed4c5ac3c2738/train/dda929c9ee9f4feb8eb8692eb5d1af7d/output/model | 172.80.10.2       | {'dim': 300, 'lr': 1.0, 'word_ngrams': 1} |
|  7 |      -0.8974 |           0.8974 |     0.218359 | ['neg', 'pos'] | benchmark_data/model/FastText/cb2018065d7e427d803b481b1e7e17d2/train/4ef1c944998e442aa1975e36e8217a2f/output/model | 172.80.10.2       | {'dim': 300, 'lr': 1.0, 'word_ngrams': 2} |
```
Metrics:
--------
Weighted F1 Score: 0.8884791256763455
Weighted Precision Score: 0.8884921831148624
Weighted Recall Score: 0.88848
Accuracy: 0.88848

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.89      0.89      0.89     12500
         pos       0.89      0.89      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.89      0.89     25000
weighted avg       0.89      0.89      0.89     25000


```

![Results](FastText/plot.png)
---
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
# Results: XLNet
```
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Build finished in 0.67 sec.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Training finished in 2395.51 sec.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:  Validation loss: 0.017671398958563806
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.885
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:  Training loss: 0.003449625564739108
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=710)[0m INFO:gobbli.experiment.base:Prediction finished in 205.30 sec.
[2m[36m(pid=710)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=710)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                              |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:--------------------------------------------------------------------------|
|  0 |    0.0176714 |            0.885 |   0.00344963 | ['neg', 'pos'] | benchmark_data/model/Transformer/7f725266c5c645ff9aae2cd91064e0b0/train/1b7e91e576da4acda9f7eb091e841dd1/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLNet', 'transformer_weights': 'xlnet-base-cased'} |
```
Metrics:
--------
Weighted F1 Score: 0.8893179724823471
Weighted Precision Score: 0.8893485289683722
Weighted Recall Score: 0.88932
Accuracy: 0.88932

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.89      0.89      0.89     12500
         pos       0.89      0.89      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.89      0.89     25000
weighted avg       0.89      0.89      0.89     25000


```

![Results](XLNet/plot.png)
---
# Results: RoBERTa
```
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Build finished in 0.23 sec.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Training finished in 1828.11 sec.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:  Validation loss: 0.017278218799829482
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8828
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:  Training loss: 0.006217972612846643
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Build finished in 0.38 sec.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=849)[0m INFO:gobbli.experiment.base:Prediction finished in 134.85 sec.
[2m[36m(pid=849)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=849)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                            |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------|
|  0 |    0.0172782 |           0.8828 |   0.00621797 | ['neg', 'pos'] | benchmark_data/model/Transformer/028e12ba75924423a3e0f545ea2184ca/train/5da16b506b084ee0b691695f9d0882b4/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Roberta', 'transformer_weights': 'roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.8888282515474982
Weighted Precision Score: 0.8890044374661773
Weighted Recall Score: 0.88884
Accuracy: 0.88884

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.90      0.88      0.89     12500
         pos       0.88      0.90      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.89      0.89     25000
weighted avg       0.89      0.89      0.89     25000


```

![Results](RoBERTa/plot.png)
---
# Results: DistilBERT
```
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.20 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 1138.47 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.01951769427470863
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8794
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.0008909900698810815
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.42 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Training finished in 1309.98 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation loss: 0.022120918272435664
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8656
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:  Training loss: 0.0011140721491072327
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Build finished in 0.45 sec.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=47)[0m INFO:gobbli.experiment.base:Prediction finished in 135.92 sec.
[2m[36m(pid=47)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=47)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                                          |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:------------------------------------------------------------------------------------------------------|
|  0 |    0.0195177 |           0.8794 |   0.00089099 | ['neg', 'pos'] | benchmark_data/model/Transformer/435c67bafd9e46c58ea9a0b422237ff3/train/9e63350a7b5d4d70aa3ba59db2169ebf/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased'}                 |
|  1 |    0.0221209 |           0.8656 |   0.00111407 | ['neg', 'pos'] | benchmark_data/model/Transformer/a08680cf48814518a3482ede744a5f93/train/725dccf1633846598aa884efeb9132d7/output/checkpoint | 172.80.10.2       | {'transformer_model': 'DistilBert', 'transformer_weights': 'distilbert-base-uncased-distilled-squad'} |
```
Metrics:
--------
Weighted F1 Score: 0.8688310072637285
Weighted Precision Score: 0.8689411760669009
Weighted Recall Score: 0.86884
Accuracy: 0.86884

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.88      0.86      0.87     12500
         pos       0.86      0.88      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000


```

![Results](DistilBERT/plot.png)
---
# Results: ALBERT
```
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Build finished in 0.25 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Training finished in 1534.16 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation loss: 0.018503880481421948
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8532
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Training loss: 0.00325197589895688
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Build finished in 0.38 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Training finished in 1561.44 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation loss: 0.01970496486723423
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8652
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:  Training loss: 0.002900374274421483
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Build finished in 0.22 sec.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=208)[0m INFO:gobbli.experiment.base:Prediction finished in 128.00 sec.
[2m[36m(pid=208)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=208)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                             |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------|
|  0 |    0.0185039 |           0.8532 |   0.00325198 | ['neg', 'pos'] | benchmark_data/model/Transformer/d04b1868e89747efadba75150e8ce7cb/train/b96fc714aa374cf6b3e860c6dbda22f8/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v1'} |
|  1 |    0.019705  |           0.8652 |   0.00290037 | ['neg', 'pos'] | benchmark_data/model/Transformer/8ea140e7b6ff414ea7a1901498317535/train/1ea5925dece04ceeb6701fd10127c8e8/output/checkpoint | 172.80.10.2       | {'transformer_model': 'Albert', 'transformer_weights': 'albert-base-v2'} |
```
Metrics:
--------
Weighted F1 Score: 0.8512799048191391
Weighted Precision Score: 0.8512808992791021
Weighted Recall Score: 0.85128
Accuracy: 0.85128

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.85      0.85      0.85     12500
         pos       0.85      0.85      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000


```

![Results](ALBERT/plot.png)
---
# Results: XLM-RoBERTa
```
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Build finished in 0.18 sec.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:Training finished in 1672.08 sec.
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:  Validation loss: 0.015142943879961967
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.7912
[2m[36m(pid=1285)[0m INFO:gobbli.experiment.base:  Training loss: 0.013316278763115407
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Build finished in 0.17 sec.
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1284)[0m INFO:gobbli.experiment.base:Prediction finished in 132.19 sec.
[2m[36m(pid=1284)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1284)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                 | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |    0.0151429 |           0.7912 |    0.0133163 | ['neg', 'pos'] | benchmark_data/model/Transformer/1b57d60374a340e0a942151f161a6e83/train/aae6c048fbee4891a21fcfc18f4fc18f/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLMRoberta', 'transformer_weights': 'xlm-roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.7942145206191481
Weighted Precision Score: 0.7949246776963078
Weighted Recall Score: 0.79432
Accuracy: 0.79432

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.78      0.82      0.80     12500
         pos       0.81      0.77      0.79     12500

    accuracy                           0.79     25000
   macro avg       0.79      0.79      0.79     25000
weighted avg       0.79      0.79      0.79     25000


```

![Results](XLM-RoBERTa/plot.png)
---
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
# Results: spaCy
```
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:Build finished in 12.91 sec.
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=527)[0m /code/gobbli/model/spacy/model.py:174: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=527)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:Training finished in 475.22 sec.
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:  Validation loss: 1.4586578167552488e-12
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.882999999998234
[2m[36m(pid=527)[0m INFO:gobbli.experiment.base:  Training loss: 2.33840203393072e-06
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Build finished in 174.79 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=525)[0m /code/gobbli/model/spacy/model.py:174: UserWarning: The spaCy model doesn't batch validation data, so the validation batch size parameter will be ignored.
[2m[36m(pid=525)[0m   "The spaCy model doesn't batch validation data, so the validation "
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Training finished in 373.63 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation loss: 1.3960288036685143e-12
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8875999999982248
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Training loss: 2.4568754242466182e-06
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Build finished in 1.04 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Training finished in 620.37 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation loss: 1.5302036970865628e-12
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8745999999982508
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Training loss: 3.398328638468229e-06
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Build finished in 0.39 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Training finished in 512.15 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation loss: 1.2428309048573283e-12
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8981999999982037
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Training loss: 2.7608190358705542e-06
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Build finished in 0.56 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Training finished in 635.63 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation loss: 1.2529507209535496e-12
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8983999999982032
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Training loss: 2.1401459845151294e-06
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Build finished in 0.34 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Training finished in 517.31 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation loss: 1.249463377206439e-12
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.8975999999982048
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:  Training loss: 1.985827455712297e-06
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Build finished in 0.69 sec.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=525)[0m INFO:gobbli.experiment.base:Prediction finished in 67.17 sec.
[2m[36m(pid=525)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=525)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | labels         | checkpoint                                                                                                                | node_ip_address   | model_params                                              |
|---:|-------------:|-----------------:|-------------:|:---------------|:--------------------------------------------------------------------------------------------------------------------------|:------------------|:----------------------------------------------------------|
|  0 |  1.45866e-12 |           0.883  |  2.3384e-06  | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/ec86dae672aa48a88a5a69732c17f694/train/15e8835d9b194ace9060ee925e0227c6/output/checkpoint | 172.80.10.2       | {'architecture': 'bow', 'model': 'en_core_web_sm'}        |
|  1 |  1.39603e-12 |           0.8876 |  2.45688e-06 | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/60e88751d9a0432aa5d0fa92f814898e/train/23a35954c8f7491e9cb914b56a69502c/output/checkpoint | 172.80.10.2       | {'architecture': 'bow', 'model': 'en_core_web_lg'}        |
|  2 |  1.5302e-12  |           0.8746 |  3.39833e-06 | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/d01b28d59cb04bf3957c7f2347acc080/train/2f9ee0af8ecd4c77bb64b1976e7267b5/output/checkpoint | 172.80.10.2       | {'architecture': 'simple_cnn', 'model': 'en_core_web_sm'} |
|  3 |  1.24283e-12 |           0.8982 |  2.76082e-06 | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/4940440210004901985400ee5681b2d6/train/028716e4996b4542934641d4a813fd94/output/checkpoint | 172.80.10.2       | {'architecture': 'simple_cnn', 'model': 'en_core_web_lg'} |
|  4 |  1.25295e-12 |           0.8984 |  2.14015e-06 | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/37246c151f0b48b5a7975dd1e202e2cd/train/ef3515aafd584ed1b2436b308e871ecd/output/checkpoint | 172.80.10.2       | {'architecture': 'ensemble', 'model': 'en_core_web_sm'}   |
|  5 |  1.24946e-12 |           0.8976 |  1.98583e-06 | ['neg', 'pos'] | benchmark_data/model/SpaCyModel/a568bb1bc91f450aa14ca4d6a7fd1677/train/6be135a3e0f941b1a37bf945c04373c3/output/checkpoint | 172.80.10.2       | {'architecture': 'ensemble', 'model': 'en_core_web_lg'}   |
```
Metrics:
--------
Weighted F1 Score: 0.8982355132444846
Weighted Precision Score: 0.8983102453575749
Weighted Recall Score: 0.89824
Accuracy: 0.89824

Classification Report:
----------------------
              precision    recall  f1-score   support

         neg       0.89      0.90      0.90     12500
         pos       0.90      0.89      0.90     12500

    accuracy                           0.90     25000
   macro avg       0.90      0.90      0.90     25000
weighted avg       0.90      0.90      0.90     25000


```

![Results](spaCy/plot.png)
---
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