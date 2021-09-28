How It Works
============

The following architecture diagram shows what the ``smjumpstart`` library covers
in the ML lifecycle.

.. image:: images/smjsindustry_system.png
  :alt: An architecture diagram of SageMaker JumpStart end-to-end solutions
        and the coverage of SageMaker JumpStart Industry Python SDK

#. Use SageMaker JumpStart Industry notebooks for solutions, models, and examples.
   The notebooks include sections of using the ``smjsindustry`` library to process
   industry text data. To find the notebooks on SageMaker JumpStart, see [Add link].
   To find a preview of the example notebooks in finance, see the
   [Add link].
#. The SageMaker JumpStart Industry Python SDK helps run SageMaker
   processing jobs to process input text data into a multimodal dataset.
   You can encrypt the Amazon S3 bucket and processing containers using Amazon VPC.
#. After the processing job has completed, SageMaker copies the result from
   the processing containers to the Amazon S3 bucket.
   SageMaker terminates the processing job and its resources.
#. You can download the result from the Amazon S3 bucket to the Studio notebook kernel
   and you can start training pretrained language models,
   such as BERT and its variants.
#. You evaluate the model performance and start using the model for making prediction.
