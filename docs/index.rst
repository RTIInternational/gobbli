.. gobbli documentation master file, created by
   sphinx-quickstart on Tue Jun  4 14:50:18 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gobbli is a library designed to make experimentation and analysis using deep learning easier.  It provides a simple, uniform interface to deep learning models that abstracts away most of the complexity in terms of different input/output formats, library versions, etc.  It attempts to implement a set of common use cases with an emphasis on usability rather than performance.

gobbli is *not* designed to provide deep learning models in a production context.  Each task generally involves running a Docker container in the background and transferring a large amount of data to and from disk, which creates significant overhead.  Additionally, gobbli does not support fine-grained model-specific tuning, such as custom loss functions.  Our goal is to take the user 80% of the way to their deep learning solution as quickly as possible so they can decide whether it's worth the effort to resolve the remaining 20%.

.. toctree::
   prerequisites
   quickstart
   troubleshooting
   advanced_usage
   api
