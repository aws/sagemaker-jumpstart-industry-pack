=======================================
SageMaker JumpStart Industry Python SDK
=======================================

.. inclusion-marker-1-starting-do-not-remove

.. image:: https://img.shields.io/pypi/v/sagemaker.svg
   :target: https://pypi.python.org/pypi/sagemaker
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/sagemaker.svg
   :target: https://pypi.python.org/pypi/sagemaker
   :alt: Supported Python Versions

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

.. image:: https://readthedocs.org/projects/sagemaker/badge/?version=stable
   :target: https://sagemaker.readthedocs.io/en/stable/
   :alt: Documentation Status

SageMaker JumpStart Industry Python SDK is an open source library for data
engineering and training (with deploying) of industry-focused machine learning
models on Amazon SageMaker JumpStart. With this industry-focused SDK,
you can curate datasets, and train and deploy models.

In particular, for the financial services industry, you can use a new set of
multimodal (long-form text, tabular) financial analysis tools within Amazon
SageMaker JumpStart. With these new tools, you can enhance your tabular ML
workflows with new insights from financial text documents and help save weeks
of development time. You can easily retrieve common financial documents,
including SEC filings, and further process financial text documents with
features such as summarization and scoring for sentiment, litigiousness,
risk, and readability. In addition, you can access language models pretrained
on financial texts for transfer learning, and use example notebooks for data
retrieval, text feature engineering, multimodal classification, and regression
models. Lastly, you can access prebuilt solutions for specific use cases
(e.g., credit scoring), which are fully customizable and showcase the use of
AWS CloudFormation templates and reference architectures to accelerate your
machine learning journey.

.. inclusion-marker-1-ending-do-not-remove

For detailed documentation, including the API reference,
see ReadTheDocs [Add Link].

.. inclusion-marker-2-starting-do-not-remove

Table of Contents
-----------------

[TBD]


Installing the SageMaker JumpStart Industry Python SDK
------------------------------------------------------

The SageMaker JumpStart Industry Python SDK is released to PyPI and
can be installed with pip as follows:

.. code-block:: bash

    pip install smjsindustry

You can also install from source by cloning this repository and running
a pip install command in the root directory of the repository:

.. code-block:: bash

    git clone https://github.com/aws/sagemaker-jumpstart-industry-python-sdk.git
    cd sagemaker-jumpstart-industry-python-sdk
    pip install .

Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker JumpStart Industry Python SDK supports Unix/Linux and Mac.

Supported Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker JumpStart Industry Python SDK is tested on:

- Python 3.6
- Python 3.7
- Python 3.8

AWS Permissions
~~~~~~~~~~~~~~~

SageMaker JumpStart Industry Python SDK runs on Amazon SageMaker. As a managed service, Amazon SageMaker performs operations on your behalf
on the AWS hardware that is managed by Amazon SageMaker.
Amazon SageMaker can perform only operations that the user permits.
You can read more about which permissions are necessary in the
`Amazon SageMaker Documentation
<https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.

SageMaker JumpStart Industry Python SDK should not require any additional permissions aside from what is required for using SageMaker.
However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

Licensing
~~~~~~~~~
SageMaker JumpStart Industry Python SDK is licensed
under the Apache 2.0 License.
It is copyright Amazon.com, Inc. or its affiliates.
All Rights Reserved. The license is available at
`Apache License <http://aws.amazon.com/apache2.0/>`_.

Running Tests
~~~~~~~~~~~~~

[TBD]


Building Sphinx Docs Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the dev version of the library:

.. code-block::

    pip install -e .\[all\]

Install Sphinx and the dependencies listed in ``sagemaker-jumpstart-industry-python-sdk/docs/requirements.txt``:

.. code-block::

    pip install sphinx
    pip install -r sagemaker-jumpstart-industry-python-sdk/docs/requirements.txt

Then ``cd`` into the ``sagemaker-jumpstart-industry-python-sdk/docs`` directory and run:

.. code-block::

    make html && open build/html/index.html


.. inclusion-marker-2-ending-do-not-remove
