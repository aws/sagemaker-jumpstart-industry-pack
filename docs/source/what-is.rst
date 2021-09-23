What is the SageMaker Jumpstart Industry Python SDK
===================================================

The SageMaker Jumpstart Industry Python SDK is for financial multimodal
(tabular and long-form text) machine learning.
It provides a set of finance text analysis capabilities as follows:

- Retrieve SEC Filings from SEC EDGAR database
- Calculate NLP scoring for text data of the SEC Filings
- Summarize text data with JaccardSummarizer and KMedoidsSummarizer
- Combine text data with tabular data into a multimodal dataset
- Provide pretrained RoBERTa-SEC language models with last decades'
  S&P 500 10-K/Q filings and the English wikipedia corpus

The SageMaker Jumpstart Industry Python SDK is launched as part of SageMaker Jumpstart.
It consists of the following deliverables:

- The SageMaker Jumpstart Industry Python SDK, i.e., `smjsindustry`
- 3 Jumpstart example notebooks
    - SEC filing retrieval, NLP scoring and summarization
    - Paycheck protection program loan return classification
    - SEC standard industry code (SIC) multi-class classification.
- 4 RoBERTa-SEC text embedding model cards in Jumpstart model zoo
- 1 JumpStart solution for corporate credit rating solution
