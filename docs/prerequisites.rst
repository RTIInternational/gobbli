Prerequisites
=============

gobbli requires Python 3.7+.

First, ensure `Docker <https://www.docker.com/>`__ is installed and your user has permissions to run docker commands.  Next, install the ``gobbli`` package and dependencies into your environment:

.. code-block:: bash

  pip install gobbli

Some of the :ref:`data-augmentation` methods require extra packages.  You can install them all using the following steps:

.. code-block:: bash

    pip install gobbli[augment]
    python -m spacy download en_core_web_sm

Additionally, :ref:`document-windowing` with the `SentencePiece <https://github.com/google/sentencepiece>`__ tokenizer requires extra packages.  Install them like so:

.. code-block:: bash

    pip install gobbli[tokenize]

.. _interactive-app-prereqs:

The `Streamlit <https://streamlit.io>`__-based :ref:`interactive-apps` require their own set of dependencies:

.. code-block:: bash

   pip install gobbli[interactive]

If you want to train models using a GPU, you will additionally need an NVIDIA graphics card and `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__.
