Finance
=======

This tutorial section shows previews of SageMaker JumpStart example notebooks
that demonstrate how to use the SageMaker JumpStart Industry
Python SDK, how to run processing jobs for loading finance documents, parsing texts,
computing scores based on NLP score types and corresponding word lists,
and creating a multimodal (TabText) dataset.
Using the processed and enhanced multimodal dataset, you'll learn
how to fine-tune pretrained BERT language models and deploy them to make predictions.

.. note::

   The SageMaker JumpStart Industry example notebooks
   are hosted and runnable only through SageMaker Studio.
   Log in to the `SageMaker console
   <https://console.aws.amazon.com/sagemaker>`_,
   and launch SageMaker Studio.
   For instructions on how to access the notebooks, see
   `SageMaker JumpStart <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html>`_ and
   `SageMaker JumpStart Industry <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart-industry.html>`_
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
