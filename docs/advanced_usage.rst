Advanced Usage
==============

gobbli provides some additional features for further customization.

Filesystem Organization
-----------------------

gobbli persists lots of data to disk, including datasets, model weights, and output.  All data will be saved under the gobbli directory, which is ``~/.gobbli`` by default.  If you want to keep your gobbli data elsewhere, set the ``GOBBLI_DIR`` environment variable to somewhere different.
    
The default directory hierarchy isolates models and task runs using unique directories named by UUIDs, which aren't particularly readable.  If you need more control over the directory hierarchy, you can use the ``data_dir`` argument when creating a model: ::

  from gobbli.model import BERT
  from pathlib import Path

  clf = BERT(
      data_dir=Path("./my_bert/")
  )
    
This will override the default organization and place all model data under the given directory, which must be empty.  For a given task (training/prediction/embedding/etc), you can also supply a user-provided name to replace the UUID: ::

    clf.train(train_input, train_dir_name='train_batch_128')

The above will store all training input and output files in a directory named ``train_batch_128`` under the model's ``data_dir``.

The hierarchy generally looks something like this:

.. code-block:: none

   GOBBLI_DIR/model/<model_class_name>/<model_data_dir_name>/<task_name>/<task_data_dir_name>/{input,output}/

For example:

.. code-block:: none

   GOBBLI_DIR/model/BERT/my_bert/train/train_batch_128/{input,output}/



.. _advanced-experimentation:

Advanced Experimentation
------------------------

gobbli experiments are limited to a predetermined workflow but include some options for customization.

- **Parameter Tuning**: Experiments accept a :paramref:`param_grid <gobbli.experiment.base.BaseExperiment.params.param_grid>` option that enables users to pass a parameter grid specifying a set of different parameters to try.  The grid should be a dictionary with parameter names (strings) as keys and lists of parameter settings to try as values.  Each parameter combination will be trained on the training set and evaluated on the validation set, and the best combination will be retrained on the combined training/validation set and evaluated on the test set for the final results.
- **Parallel/Distributed Experimentation**: gobbli uses `ray <https://ray.readthedocs.io/en/latest/>`__ under the hood to run multiple training/validation steps in parallel.  Ray creates and uses a local cluster composed of all CPUs on your machine by default, but it can also be used to add GPUs or connect to an existing distributed cluster. Note ray (and gobbli) must be installed on all worker nodes in the cluster.  Experiments accept an optional :paramref:`ray_kwargs <gobbli.experiment.base.BaseExperiment.params.ray_kwargs>` option, which is passed directly to :func:`ray.init`.  Use this parameter for more control over the underlying Ray cluster.  **NOTE:** If you're running an experiment on a single node, gobbli will simply pass checkpoints around as file paths, since the Ray master and workers share a filesystem.  If you're running a distributed experiment, gobbli cannot rely on file paths being the same between workers and the master node, so it will save checkpoints as gzip-compressed tar archives in memory and store them in the Ray object store.  This means your object store must be able to hold weights for as many trials as will be run in one experiment, which may be a **lot** of memory.
- **Enabling GPU support**: During experiments, gobbli exposes GPUs to models based on whether they're made available to the Ray cluster and are required for tasks.  To run a GPU-enabled experiment, reserve a nonzero number of GPUs for each task via the :paramref:`task_num_gpus <gobbli.experiment.base.BaseExperiment.params.task_num_gpus>` parameter and tell Ray the cluster contains a nonzero number of GPUs via the :obj:`num_gpus` argument to :func:`ray.init`.

Metadata
--------

Each model and task write JSON-formatted metadata to their respective data directories containing parameters and other useful information.  The metadata can be read to recall what parameters were used to train a given model, where the checkpoint for a training task is stored, how many embeddings were generated, etc.

Model metadata is stored in the model's data directory in a file named ``gobbli-model-meta.json``.  The metadata generally contains model parameters that can be used to recreate the same model later (see `Re-Initializing Models`_).  See the :meth:`init()` method for classes derived from :class:`gobbli.model.base.BaseModel` for more info on which keys should be expected in the metadata.  Example model metadata:

.. code-block:: json

   {
       "max_seq_length": 128
   }

Task metadata is stored in the task's directory in a file named ``gobbli-task-meta.json``.  For input tasks, the metadata generally contains the task parameters and some summary information about the input.  For output tasks, the metadata usually has the locations of any generated artifacts and summary information about the generated output.  See the :meth:`metadata()` method for classes derived from :class:`gobbli.io.TaskIO` for more info on which keys should be expected in the metadata.  Example task metadata:

.. code-block:: json

  {
      "train_batch_size": 32,
      "valid_batch_size": 8,
      "num_train_epochs": 1,
      "len_X_train": 40,
      "len_y_train": 40,
      "len_X_valid": 10,
      "len_y_valid": 10
  }

Re-Initializing Models
-----------------------
    
You can re-initialize a model from the metadata in an existing data directory using the ``load_existing`` argument -- the model will reload its parameters from the metadata file in that directory, so you don't have to specify them again.  To reload the model created with non-default parameters above in a different session: ::

  clf = BERT(
      data_dir=Path("./my_bert/"),
      load_existing=True
  )

Running gobbli Inside a Docker Container
----------------------------------------

Since gobbli must spawn its own Docker containers, there are some extra complications when trying to run it from inside a Docker container (as opposed to natively on the host machine).

 - You must mount ``/var/run/docker.sock`` on the host to the same directory on the container.  This is needed to allow the Docker client in the container to communicate with the daemon on the host.
 - Any directories that should contain persistent files (your gobbli directory, custom model directories, etc.) which themselves need to be mounted in spawned containers must be mounted in the main Docker container **with the same name they would have on the host**.  This is because the Docker daemon can only see paths on the host, so any paths that need to be mounted in containers must also exist on the host.  You can accomplish this with something like the following mount declaration: ``$(pwd):$(pwd)``.

See the ``gobbli-ci`` service declaration in ``ci/docker-compose.yml`` for a working example of how to properly run gobbli inside a Docker container.
