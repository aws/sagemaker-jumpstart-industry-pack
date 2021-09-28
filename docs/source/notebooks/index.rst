Finance
=======

The following example notebooks show how to use the SageMaker JumpStart Industry
Python SDK to run processing jobs for loading finance documents, parsing texts,
computing scores based on NLP score types, and creating a multimodal (TabText) dataset.
Using the processed and enhanced multimodal dataset, you'll learn
how to train BERT language models and deploy them to make predictions.

.. note::

   The SageMaker JumpStart Industry example notebooks
   are hosted and runnable only through SageMaker Studio.
   Log in to the `SageMaker console
   <https://console.aws.amazon.com/sagemaker>`_,
   and launch SageMaker Studio.
   To find the instructions on how to access the notebooks, see
   `SageMaker JumpStart <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html>`_
   in the *Amazon SageMaker Developer Guide*.

.. important::

   The example notebooks are for demonstrative purposes only.
   The notebooks are not financial advice and should not be relied on as
   financial or investment advice.


.. toctree::
   :maxdepth: 2

   finance/notebook1/SEC_Retrieval_Summarizer_Scoring
   finance/notebook2/PPP_TabText_ML
   finance/notebook3/SEC_MNIST_ML
