# ERROR: Exception during run 'GPT'.

```
  File "/data/users/jnance/gobbli/benchmark/scenario.py", line 153, in run
    output = self._do_run(run, run_output_dir)

  File "/data/users/jnance/gobbli/benchmark/scenario.py", line 430, in _do_run
    run_kwargs=run.run_kwargs,

  File "/data/users/jnance/gobbli/benchmark/benchmark_util.py", line 214, in run_benchmark_experiment
    return exp.run(**run_kwargs)

  File "/code/gobbli/experiment/classification.py", line 659, in run
    for params in grid

  File "/usr/local/lib/python3.7/site-packages/ray/worker.py", line 2247, in get
    raise value

RayTaskError: [36mray_worker:gobbli.experiment.classification.train()[39m (pid=6724, host=0dd5b0b5e4f4)
  File "/code/gobbli/experiment/classification.py", line 550, in train
    train_output = clf.train(train_input)
  File "/code/gobbli/model/mixin.py", line 104, in train
    _run_task(self._train, train_input, self.train_dir(), train_dir_name),
  File "/code/gobbli/model/mixin.py", line 42, in _run_task
    task_output = cast(gobbli.io.TaskIO, task_func(task_input, context))
  File "/code/gobbli/model/transformer/model.py", line 254, in _train
    self.docker_client, self.image_tag, cmd, self.logger, **run_kwargs
  File "/code/gobbli/docker.py", line 113, in run_container
    raise RuntimeError(err_msg)
RuntimeError: Error running container (return code 1). Last 20 lines of logs: 
Using device: cuda
Number of GPUs: 1
Initializing transformer...
[36mray_worker:gobbli.experiment.classification.train()[39m (pid=6724, host=0dd5b0b5e4f4)
  File "run_model.py", line 461, in <module>
  Model: GPTForSequenceClassification
    Weights: openai-gpt
  Tokenizer: GPTTokenizer
  Config: GPTConfig
    model_cls = getattr(transformers, model_name)
AttributeError: module 'transformers' has no attribute 'GPTForSequenceClassification'


```