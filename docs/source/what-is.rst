What Is the SageMaker JumpStart Industry Python SDK
===================================================

SageMaker JumpStart Industry Python SDK is an open source client library for
processing text datasets, training ML language models such as BERT and its variants,
and deploying industry-focused
machine learning models on Amazon SageMaker JumpStart. With this
industry-focused SDK, you can curate datasets, and train and deploy models.

.. include:: ../../README.rst
   :start-after: inclusion-marker-1-1-starting-do-not-remove
   :end-before:  inclusion-marker-1-1-ending-do-not-remove

The library is to process financial multimodal
(tabular and long-form text) datasets for machine learning.
It provides a set of finance text analysis capabilities as follows:

- Retrieve SEC Filings from SEC EDGAR database.
- Calculate NLP scores for text data of the SEC Filings.
- Summarize text data with JaccardSummarizer and KMedoidsSummarizer.
- Combine text data with tabular data into a multimodal dataset.
- Provide pretrained RoBERTa-SEC language models with
  S&P 500 10-K/Q filings over the last decades and the English wikipedia corpus.

The SageMaker JumpStart Industry Python SDK is a client library for SageMaker JumpStart.
It consists of the following deliverables:

- The SageMaker JumpStart Industry Python SDK, i.e., `smjsindustry`
- 3 JumpStart example notebooks on SageMaker Studio

  - SEC filing retrieval, NLP scoring and summarization
  - Paycheck protection program loan return classification
  - SEC standard industry code (SIC) multi-class classification.

- 4 RoBERTa-SEC text embedding model cards in JumpStart model zoo
- 1 JumpStart solution for corporate credit rating solution
