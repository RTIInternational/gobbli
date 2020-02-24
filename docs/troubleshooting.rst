Troubleshooting
===============

gobbli uses some heavy-duty abstractions to hide a lot of complexity, so there's a lot of potential for complex errors along the way.  Here are some things you might run into:

My model is predicting only one class
-------------------------------------

This indicates something's wrong with the training process.  Some things to check:

- Do you have a very imbalanced dataset?  You may need to raise the batch size, if possible, or try downsampling/upsampling to ensure you don't get many batches composed of only one class.
- Is your dataset ordered by label?  The model needs to have a mix of classes in each batch to learn effectively.  Ensure your training dataset is shuffled.  gobbli :class:`Datasets <gobbli.dataset.base.BaseDataset>` take care of this with the :meth:`train_input <gobbli.dataset.base.BaseDataset.train_input>` method.

I'm running out of CPU memory
-----------------------------

Your dataset might be too big to fit in memory.  gobbli currently doesn't support lazily loading datasets from disk, so anything you use to train has to fit in memory.  You can try:

 - Sampling from your dataset.
 - If you're running an experiment in parallel, try reducing the number of models training at a time to reduce the number of copies of your dataset in memory.
 - If you're running a distributed experiment, the ray object store has to be large enough to fit weights for all models trained during the experiment at the same time.  You may need to train fewer models or increase the size of the object store.

I'm running out of GPU memory
-----------------------------

Some models are larger than others.  You can try:

 - Decreasing the :paramref:`train_batch_size <gobbli.io.TrainInput.params.train_batch_size>` if you're training; this is the biggest driver of GPU memory usage.  Beware of making the batch size so small that the model can't update gradients accurately, though. The :class:`gobbli.model.transformer.Transformer` model implements gradient accumulation, which can be used to counteract the detrimental effect of a smaller batch size.
 - Decreasing the ``max_seq_len`` parameter of your model, if it has one.  Consider using :ref:`document-windowing` if you do this to account for the truncation of your texts.
 - Using a smaller set of pretrained weights (ex. instead of ``bert-large-uncased``, try ``bert-base-uncased``).

A gobbli function appears to be hanging
---------------------------------------

gobbli sometimes needs to download very large files in the background -- for example, the pretrained weights for BERT Large are over 1GB.  It can look like nothing is happening while this is going on.  You can try enabling debug logs to see (much) more detailed information about what gobbli is doing behind the scenes: ::

    import logging
    logging.basicConfig(level=logging.DEBUG)

We have noticed that sometimes a download can time out repeatedly, depending on the quality of your internet connection.  If this keeps happening, you can try manually downloading the file gobbli is trying to download by pasting the URL in your browser and moving the resulting file to the path where gobbli is trying to save it.  gobbli should detect the file and not attempt to redownload the file the next time it tries to access the file.
