Quickstart
==========

Models
------

Since deep learning models can take a long time to train, gobbli provides trivial models for certain use cases that can be used to verify your code runs properly without investing a long time into model training.

- Majority classifier (:class:`gobbli.model.majority.MajorityClassifier`): Classifies each example as the most frequent class in the training set.  Supports training and prediction.
- Random embedder (:class:`gobbli.model.random.RandomEmbedder`): Generates a fixed-size random length vector for each document.  Supports embedding generation.

We recommend scaffolding your code using one of these techniques before switching to a real model.  Here are the currently implemented models:

- `Google's BERT <https://github.com/google-research/bert>`__: (:class:`gobbli.model.bert.BERT`) Supports training/prediction (multiclass) and embedding generation.
- `Microsoft's MT-DNN <https://github.com/namisan/mt-dnn>`__: (:class:`gobbli.model.mtdnn.MTDNN`) Supports training/prediction (multiclass).
- `Universal Sentence Encoder (USE) <https://tfhub.dev/google/universal-sentence-encoder/2>`__: (:class:`gobbli.model.use.USE`) Supports embedding generation.
- `Facebook's fastText <https://github.com/facebookresearch/fastText>`__: (:class:`gobbli.model.fasttext.FastText`) Supports training/prediction (multiclass and multilabel) and embedding generation.
- `transformer models <https://github.com/huggingface/transformers>`__: (:class:`gobbli.model.transformer.Transformer`) Models with a ``<model>ForSequenceClassification`` version implemented can be used for training/prediction (multiclass and multilabel) and embedding generation (ex. ``Bert``).  All other models can only be used for embedding generation.
- `scikit-learn models <https://scikit-learn.org/stable/>`__: (:class:`gobbli.model.sklearn.SKLearnClassifier`) Any scikit-learn pipeline which accepts text input and outputs a predicted probability can be used as a gobbli model.  A simple default is implemented composing TF-IDF vectorization and logistic regression.  Baseline "embeddings" are also provided via a TF-IDF vectorizer (:class:`gobbli.model.sklearn.TfidfEmbedder`).  Multilabel classification is supported by wrapping the passed classifier in a :class:`sklearn.multiclass.OneVsRestClassifier`.
- `spaCy models <https://spacy.io/>`__: (:class:`gobbli.model.spacy.SpaCyModel`) The text categorizer component of any spaCy language model (or spacy-transformers model) can be trained and used for prediction (multiclass and multilabel).  The spaCy model vectors can also be retrieved as static embeddings (pre-training not supported).

Most models can accept model-specific parameters during initialization.  See the documentation for each model's :meth:`init` method for information on model-specific parameters.

Logging
-------

gobbli implements some logging with timing for deep learning models.  If you want to see some more detailed information while running tasks, set up logging like so: ::

  import logging
  logging.basicConfig(level=logging.INFO)

Using ``level=logging.DEBUG`` will directly propagate logs from any spawned Docker containers for even more detailed status information.

High-Level API -- Experiments
-----------------------------

gobbli's high-level API supports canned experimentation workflows based on a couple of tasks.  It's easiest to start out with an Experiment and drop down to the lower-level Task API if you need more flexibility.

A high-level overview of each type of experiment follows.  For an overview of more detailed configuration options, including parameter tuning, parallel/distributed experiments, and using GPUs in experiments, see :ref:`advanced-experimentation`.  For example experiments on benchmark datasets, see the Markdown documents in the ``benchmark/`` directory of the repository.

Classification Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^

This type of experiment is useful when you have a classification problem, or a set of documents with associated labels.  A :class:`gobbli.experiment.classification.ClassificationExperiment` requires a model and dataset.  The dataset can be either a :class:`gobbli.dataset.base.BaseDataset` derived class instance or an (X, y) tuple, where X is a list of strings, and y is a corresponding list of labels (or, for multilabel classification, a list of lists of labels).  The dataset will be split into train, validation, and test sets.  Training will be run on the train set, evaluated on the validation set, and results on the test set will be reporting.

To run an experiment: ::

  from gobbli.experiment import ClassificationExperiment
  from gobbli.model import MajorityClassifier

  X = [
      "This is positive.",
      "This is negative.",
      "This is bad.",
      "This is good.",
      "This is really bad.",
      "This is really good.",
      "This is pretty good.",
      "This is pretty bad.",
  ]

  y = [
      "Good",
      "Bad",
      "Bad",
      "Good",
      "Bad",
      "Good",
      "Good",
      "Bad",
  ]

  exp = ClassificationExperiment(
    model_cls=MajorityClassifier,
    dataset=(X, y)
  )

  results = exp.run()

The results object contains raw output (including predicted probabilities) on the test set and several methods for analyzing it, including metrics, error analysis, and plotting.  If the provided methods aren't sufficient, you can perform your own results analysis using the raw output.  See :class:`gobbli.experiment.classification.ClassificationExperimentResults` for more information.

If you want to reuse the resulting model checkpoint in the future, use the :meth:`get_checkpoint <gobbli.experiment.classification.ClassificationExperimentResults.get_checkpoint>` method to save the checkpoint to your filesystem.  The returned path can be directly passed to future invocations of the model class to make more predictions or continue training.

Low-Level API - Tasks
---------------------

If you require more specialized workflows, you can use the lower-level Task API.  Experiments run a canned set of tasks with some sensible default parameters.  See the following sections for more information on the individual tasks each experiment is composed of.

Training
^^^^^^^^

Deep learning models can generally be fine-tuned on a user's specific problem after having been pretrained on a large, general dataset.  Training enables the model to develop an internal representation more suited to the nuances of a given problem.  We generally train models in a classification paradigm, encouraging them to learn to predict a set of labels.

Most gobbli models can be trained. First, create your training input. Multilabel classification is also transparently supported; just pass a list of lists of labels instead of a list containing a single label for each document. ::

  from gobbli.io import TrainInput

  train_input = TrainInput(
      # X_train: A list of strings to classify
      X_train=["This is a training document.",
               "This is another training document."],
      # y_train: The true class for each string in X_train
      y_train=["0", "1"],
      # And likewise for validation
      X_valid=["This is a validation sentence.",
               "This is another validation sentence."],
      y_valid=["1", "0"],
      # Number of documents to train on at once
      train_batch_size=1,
      # Number of documents to evaluate at once
      valid_batch_size=1,
      # Number of times to iterate over the training set
      num_train_epochs=1
  )

Now set up your model. ::

  from gobbli.model import MajorityClassifier

  clf = MajorityClassifier()

  # Set up classifier resources -- Docker image, etc.
  clf.build()

Finally, train the model and inspect the output, if you want. See :class:`gobbli.io.TrainOutput` for the supported properties. ::
  
  train_output = clf.train(train_input)

Training is generally used to enhance performance on other tasks, such as classification or generating embeddings, rather than being the end product itself.

Predicting
^^^^^^^^^^^

Classification models predict whether the input falls into one of several predetermined classes (or, for a multilabel model, which of several labels apply).

With a trained model, we can make predictions. ::

  from gobbli.io import PredictInput

  predict_input = PredictInput(
      # X: A list of strings to predict the trained classes for
      X=["Which class is this document?"],
      # Pass the set of labels, trained checkpoint, and
      # whether the model was multilabel from the training output
      labels=train_output.labels,
      checkpoint=train_output.checkpoint,
      multilabel=train_output.multilabel,
      # Number of documents to predict at once
      predict_batch_size=1
  )

  predict_output = clf.predict(predict_input)

See :class:`gobbli.io.PredictOutput` for the output structure.
    

Generating Embeddings
^^^^^^^^^^^^^^^^^^^^^

A document embedding is a numeric vector representing the semantic meaning of a document.  Embeddings can be used in place of simpler word counts/TF-IDF vectorization methods to account for problems like synonyms having similar meanings despite using different words.  The resulting vectors can be used for applications like determining similarity between documents and/or clustering.

Embeddings can be generated from a trained model.  Some models also use pretrained weights that can provide a decent representation of documents without additional training.  In their case, training is optional but may improve the results.

An example of generating embeddings: ::

  from gobbli.model import RandomEmbedder
  from gobbli.io import EmbedInput

  clf = RandomEmbedder()
  clf.build()

  # Construct input for embedding generation
  embed_input = EmbedInput(
      # X: A list of strings to generate embeddings for
      X=["We want an embedding for this.", "Also for this."],
      # Number of documents to generate embeddings for at once
      embed_batch_size=1,
      # How to pool the token embeddings to generate a document embedding
      pooling=gobbli.io.EmbedPooling.MEAN,
      checkpoint=train_output.checkpoint
  )

  embed_output = clf.embed(embed_input)
    
See :class:`gobbli.io.EmbedOutput` for the output structure.

Interactive Apps
----------------

Now that you understand the basics of how gobbli works, you might want to try out some of gobbli’s :ref:`interactive-apps` to perform some common tasks without writing any code.


Extras
------

gobbli provides some additional functionality that can be used with or independently of its models. If you want to use gobbli to augment your dataset and transfer the dataset to another modeling framework, feel free.


.. _data-augmentation:

Data Augmentation
^^^^^^^^^^^^^^^^^

gobbli provides some helper functions to perform data augmentation.  If you only have a small set of labeled data, generating new data can help your model perform better.  Generated data will be similar but not exactly equal to the original data (similarity can generally be tweaked using some parameters), so you can apply your existing labels to the new data.

gobbli currently implements 3 data augmentation strategies, listed below.  All methods allow you to configure the proportion of words flagged for replacement and the amount of data generated.

- :class:`gobbli.augment.word2vec.Word2Vec`: Generate new documents by tokenizing existing documents and replacing a subset of tokens with similar words according to a Word2Vec model.  We use `gensim's word2vec implementation <https://radimrehurek.com/gensim/models/word2vec.html>`__ under the hood, so this method requires `installing gensim <https://radimrehurek.com/gensim/install.html>`__.  You can pick one of several pretrained gensim word2vec models or supply your own.  Tokenization can be naive, spaCy-based (requires `installing spaCy <https://spacy.io/usage>`__), or custom.  See the class documentation for additional configuration options.
- :class:`gobbli.augment.wordnet.WordNet`: Generate new documents by part-of-speech tagging existing documents (requires `installing spaCy <https://spacy.io/usage>`__) and replacing a subset of tokens with synonyms/hypernyms/hyponyms according to the `WordNet lexical database <https://wordnet.princeton.edu/>`__ (requires `installing nltk <https://www.nltk.org/install.html>`__). You can configure the language model used by spaCy to do tagging.
- :class:`gobbli.augment.bert.BERTMaskedLM`: Generate new documents using the language modeling capabilities of `BERT <https://github.com/google-research/bert>`__, as implemented in `transformers <https://github.com/huggingface/transformers>`__.  The model predicts each masked word using the surrounding context, generating new documents.  You can use any pretrained BERT model supported by pytorch-transformers.  See the class documentation for additional configuration options.

An example of augmenting a dataset: ::

  from gobbli.augment import WordNet

  wn = WordNet()

  X = ["This is positive.", "This is negative."]
  y = ["1", "0"]

  times = 3
  X_augmented = X + wn.augment(X, times=times, p=0.5)
  y_augmented = y + (y * times)

.. _document-windowing:

Document Windowing
^^^^^^^^^^^^^^^^^^

Many advanced deep learning models have a fixed max sequence length to limit memory usage for long documents.  If you don't have enough memory available to raise the sequence length to fit all your documents, you can use gobbli's "document windowing" helpers.

The idea is to tokenize each document and split it into equal-length windows roughly equal to your model's max sequence length, which will prevent your model from missing any of the information in the documents during training.  For tasks after training (such as prediction and embedding), the windowed output can then be pooled in a way that makes sense for your problem.  For example, if you're generating embeddings, you probably want each document embedding to be the mean of all the windows, but if you're building a classifier to detect whether a subject is discussed in a document, you may want the output predicted probability for each class to be the maximum of all the windows.

You'll want to use the :class:`gobbli.util.TokenizeMethod` most similar to your model's tokenizer to get the most precise windowing.

Here's an example of document windowing: ::

  from gobbli.io import make_document_windows, pool_document_windows
  from gobbli.util import TokenizeMethod

  X = ["This is a long sentence.", "This is short."]
  y = ["1", "0"]

  # Convert the documents to windows
  X_windowed, X_windowed_indices, y_windowed = make_document_windows(X, 3, y=y)
  # The above objects all contain one or more rows for each window in the document

  # Get predictions or embeddings from a model
  input = PredictInput(
    X=X_windowed,
    labels=["1", "0"],
  )
  output = ... 

  # Pool the predictions for the output in-place
  pool_document_windows(output, X_windowed_indices)

  # Now you can compare the pooled predictions to the original "y"
