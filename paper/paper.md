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

Machine learning has long been used to address natural language processing (NLP) tasks like sentiment analysis [@Pang:2008] and document classification [@Aggarwal:2012].  Traditional approaches to these tasks require numerous labeled examples from the specific domains in which they will be applied.  Such algorithms can only use the available training data, which is often limited in size and/or diversity, to learn to understand natural language.  In the last few years, a concept known as transfer learning [@Weiss:2016] has caused a paradigm shift in NLP.  Rather than training distinct task-specific models from scratch, a transfer learning model first learns the rules of language from a large, diverse text corpus during an extensive self-supervised training regimen.  Self-supervised tasks are formulated such that unlabeled data can be used to train supervised models [@Raina:2007]; an example is masked language modeling, where a subset of words in a document are masked out, and the model predicts the masked words [@Devlin:2018].  The transfer learning model thus learns a rich representation of language which can be fine-tuned to solve specific problems.  According to @Torrey:2010, this approach mimics how humans reuse their general understanding of language across tasks.  Transfer learning has not only rapidly advanced the state of the art in classification but has enabled near-human performance on more advanced tasks like question answering [@Rajpurkar:2016; @Raffel:2019] and natural language inference [@Rajpurkar:2016; @Williams:2018; @Bowman:2015].

While the performance gains on benchmark tasks are undeniable, applied researchers face challenges using transfer learning models to solve new problems.  A wide variety of models are being developed by disparate research teams using different technologies [@Sun:2019; @Raffel:2019; @Liu:2019]. A practitioner may therefore be required to learn a new programming language, a deep learning library, a containerization technology, and a model interface whenever they want to evaluate feasibility of a new model on a custom task.  ``gobbli`` was developed to address this problem.

``gobbli`` is a Python library intended to bridge state-of-the-art research in natural language processing and application to real-world problems.  The library defines a simple interface for training classification models, producing predictions, and generating embeddings.  Several models implementing the interface are available using programmatically-created Docker containers to abstract away differences in underlying deep learning libraries and model hyperparameters.  This approach allows users to easily evaluate models and compare performance across model types without spending time adapting their dataset and use case to each model.  Compared to other deep learning libraries used for NLP like ``transformers`` [@Wolf:2019] and ``fastai`` [@Howard:2018], ``gobbli`` is designed to emphasize simplicity and interoperability rather than customization and performance in order to make deep learning more accessible to applied researchers.

Beyond its model wrappers, ``gobbli`` provides several helpful utilities for NLP practitioners.  Data augmentation has emerged as a popular technique to improve model performance when training data is limited.  Multiple methods for data augmentation are implemented in ``gobbli``, including backtranslation [@Shleifer:2019], word replacement [@Wei:2019], and contextual augmentation [@Kobayashi:2018]. These methods can be used independently of ``gobbli`` models for interoperability with other modeling libraries.  ``gobbli`` also bundles a set of interactive web applications built using Streamlit [@Teixeira:2018] which can be used to explore a dataset, visualize embeddings, evaluate model performance, and explain model predictions without writing any code.

``gobbli`` was developed from experiences on client contracts and internal projects at RTI International.  It is intended for use by anyone solving problems using applied NLP, including researchers, students, and industry practitioners.

# Acknowledgments

Work on ``gobbli`` was funded by RTI International.

# References
