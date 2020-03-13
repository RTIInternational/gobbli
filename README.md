<div align="center">
  <div>
    <img src="https://raw.githubusercontent.com/RTIInternational/gobbli/master/img/gobbli_lg.svg?sanitize=true" alt="gobbli logo" width="200" />
  </div>
  <div>
    <a href="https://travis-ci.com/RTIInternational/gobbli"><img src="https://travis-ci.com/RTIInternational/gobbli.svg?branch=master" alt="PyPI version"></a>
    <a href="https://badge.fury.io/py/gobbli"><img src="https://badge.fury.io/py/gobbli.svg" alt="PyPI version"></a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/gobbli" />
    <a href="https://doi.org/10.5281/zenodo.3516024"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3516024.svg" alt="DOI"></a>
  </div>
</div>

This is a library designed to provide a uniform interface to various deep learning models for text via programmatically created Docker containers.

## Usage

See [the docs](https://gobbli.readthedocs.io/en/latest/) for prerequisites, a quickstart, and the API reference.  In brief, you need [Docker](https://www.docker.com/) installed with appropriate permissions for your user account to run Docker commands and Python 3.7.  Then run the following:

    pip install gobbli

You may also want to check out the [benchmarks](./benchmark) to see some comparisons of gobbli's implementation of various models in different situations.

### Interactive

gobbli provides [streamlit](https://www.streamlit.io/) apps to perform some interactive tasks in a web browser, such as data exploration and model evaluation.  Once you've installed the library, you can run the bundled apps using the `gobbli` command line application.  Check [the docs](https://gobbli.readthedocs.io/en/latest/interactive_apps.html) for more information.

## Development

Assuming you have all prerequisites noted above, you need to install the package and all required + optional dependencies in development mode:

    pip install -e ".[augment,tokenize,interactive]"
    
Install additional dev dependencies:

    pip install -r requirements.txt
    
Run linting, autoformatting, and tests:

    ./run_ci.sh
    
To avoid manually fixing some of these errors, consider enabling [isort](https://github.com/timothycrosley/isort) and [black](https://github.com/python/black) support in your favorite editor.

If you're running tests in an environment with less than 12GB of memory, you'll want to pass the `--low-resource` argument when running tests to avoid out of memory errors.
    
**NOTE:** If running on a Mac, even with adequate memory available, you may encounter Out of Memory errors (exit status 137) when running the tests.  This is due to not enough memory being allocated to your Docker daemon.  Try going to Docker for Mac -> Preferences -> Advanced and raising "Memory" to 12GiB or more.

If you want to run the tests GPU(s) enabled, see the `--use-gpu` and `--nvidia-visible-devices` arguments under `py.test --help`.  If your local machine doesn't have an NVIDIA GPU, but you have access to one that does via SSH, you can use the `test_remote_gpu.sh` script to run the tests with GPU enabled over SSH.

### Docs

To generate the docs, install the docs requirements:

    pip install -r docs/requirements.txt
    
Since doc structure is auto-generated from the library, you must have the library (and all its dependencies) installed as well.

Then, run the following from the repository root:
    
    ./generate_docs.sh
    
Then browse the generated documentation in `docs/_build/html`.

    
## Attribution

gobbli wouldn't exist without the public release of several state-of-the-art models.  The library incorporates:

- [BERT](https://github.com/google-research/bert), released by Google
- [MT-DNN](https://github.com/namisan/mt-dnn), released by Microsoft
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2), released by Google
- [fastText](https://github.com/facebookresearch/fastText), released by Facebook
- [transformers](https://github.com/huggingface/transformers), released by Hugging Face
- [spaCy](https://spacy.io/), by Explosion
- [scikit-learn](https://scikit-learn.org/stable/)

Original work on the library was funded by [RTI International](https://www.rti.org/).

Logo design by [Marcia Underwood](http://marciaunderwood.com).
