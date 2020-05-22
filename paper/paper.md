---
title: 'gobbli: A uniform interface to deep learning for text in Python'
tags:
 - Python
 - deep learning
 - data science
 - classification
 - natural language processing
authors:
 - name: Jason Nance
   orcid: 0000-0003-4127-3198
   affiliation: 1
 - name: Peter Baumgartner
   orcid: 0000-0003-3117-6239
   affiliation: 1
affiliations:
 - name: RTI International
   index: 1
date: 20 May 2020
bibliography: paper.bib
---

# Summary

Machine learning models have long been used to tackle natural language processing (NLP) problems like sentiment analysis [@Pang:2008] and document classification [@Aggarwal:2012].  These algorithms were originally only capable of solving the narrow problems defined by their training datasets.  In the last few years, a concept known as transfer learning [@Weiss:2016] has caused a complete paradigm shift in NLP.  Instead of solving each task separately, a transfer learning model first learns the rules of language during an extensive self-supervised training regimen and then fine-tunes this rich language model for specific problems.  According to @Torrey:2010, this approach mimics how humans reuse their general understanding of language across tasks.  Transfer learning has not only rapidly advanced the state of the art in classification and other traditional tasks but has enabled near-human performance on more advanced tasks like question answering [@Rajpurkar:2016] and paraphrasing [@Dolan:2005].

While the performance gains on benchmark tasks are undeniable, applied researchers face challenges using transfer learning models to solve new problems.  A wide variety of models are being developed by disparate research teams using different technologies [@Sun:2019; @Raffel:2019; @Liu:2019]. A practitioner may therefore be required to learn a new programming language, deep learning library, and model interface whenever they want to evaluate feasibility of a new model on a custom task.  ``gobbli`` was developed to address this problem.

``gobbli`` is a Python library intended to bridge state-of-the-art research in natural language processing and application to real-world problems and data.  The library defines a simple interface for training classification models, generating predictions, and generating embeddings.  Several models fulfilling the interface are implemented using programmatically-created Docker containers to abstract away differences in underlying deep learning libraries and model parameters.  This approach allows users to easily evaluate models and compare performance across model types without spending time adapting their dataset and use case to each model.  Compared to other deep learning libraries used for NLP like ``transformers`` [@Wolf:2019] and ``fastai`` [@Howard:2018], ``gobbli`` is designed to emphasize simplicity and interoperability rather than customization and performance in order to make deep learning more accessible to applied researchers.

Beyond its model wrappers, ``gobbli`` provides several helpful utilities for NLP practitioners.  Data augmentation has emerged as a popular technique to improve model performance when training data is limited [@Wei:2019].  Multiple methods for data augmentation are implemented in ``gobbli``, including backtranslation [@Shleifer:2019], word replacement, and masked language modeling. These methods can be used independently of ``gobbli`` models for interoperability with other modeling libraries.  ``gobbli`` also bundles a set of interactive web applications built using Streamlit [@Teixeira:2018] which can be used to explore a dataset, visualize embeddings, evaluate model performance, and explain model predictions without writing any code.

``gobbli`` was developed for use on client contracts and internal projects at RTI International.  It is intended for use by anyone solving problems using applied NLP, including researchers, students, and industry practitioners.

# Acknowledgments

Work on ``gobbli`` was funded by RTI International.

# References