Classify SEC 10K/Q Filings to Industry Codes Based on the MDNA Text Column
==========================================================================

Introduction
------------

Objective
~~~~~~~~~

The purpose of this notebook is to address the following question: Can
we train a model to detect the broad industry category of a company from
the text of Management Discussion & Analysis (**MD&A**) section in SEC
filings?

This notebook demonstrates how to use of text data in U.S. Securities
and Exchange Commission (SEC) filings, matching industry codes, adding
NLP scores, and creating a *multimodal* training dataset. The multimodal
dataset is then used for training a model for *multiclass*
classification tasks.

Curating Input Data
~~~~~~~~~~~~~~~~~~~

This example notebook demonstrates how to train a model on a synthetic
training dataset that’s curated using the SEC Forms retrieval tool
provided by the SageMaker JumpStart Industry Python SDK. You’ll download
a large number of SEC 10-K/Q forms for companies in the S&P 500 from
2000 to 2019. A separate column of the dataframe contains the **MD&A**
section of the filings. The **MD&A** section is chosen because it is the
most popular section used in the finance industry for natural language
processing (NLP). The `SIC industry
codes <https://www.osha.gov/data/sic-manual>`__ are also used for
matching to those in the `NAICS
system <https://www.census.gov/naics/>`__.

   **Important**: This example notebook is for demonstrative purposes
   only. It is not financial advice and should not be relied on as
   financial or investment advice.

General Steps
~~~~~~~~~~~~~

This notebook takes the following steps: 1. Prepare training and testing
datasets. 2. Add NLP scores to the MD&A text features. 3. Train the
AutoGluon model for classification on the extended dataframe of MD&A
text and NLP scores.

SageMaker Studio Kernel Setup
-----------------------------

Recommended kernel is **Python 3 (Data Science)**. *DO NOT* use the
**Python 3 (SageMaker JumpStart Data Science 1.0)** kernel because there
are some differences in preinstalled dependency. For the instance type,
using a larger instance with sufficient memory can be helpful to
download the following materials.

Load Data, SDK, and Dependencies
--------------------------------

The following code cells download the ``smjsindustry`` SDK,
dependencies, and dataset from an S3 bucket prepared by SageMaker
JumpStart Industry. You will learn how to use the ``smjsindustry`` SDK
which contains various APIs to curate SEC datasets. The dataset in this
example was synthetically generated using the ``smjsindustry`` package’s
SEC Forms Retrieval tool.

.. code:: ipython3

    notebook_artifact_bucket = 'jumpstart-cache-alpha-us-west-2'
    notebook_data_prefix = 'smfinance-notebook-data/mnist'
    notebook_sdk_prefix = 'smfinance-notebook-dependency/smjsindustry'
    notebook_autogluon_prefix = 'smfinance-notebook-dependency/autogluon'

.. code:: ipython3

    # Download example dataset
    data_bucket = f's3://{notebook_artifact_bucket}/{notebook_data_prefix}'
    !aws s3 sync $data_bucket ./

Install packages running the following code block. It installs packages
that are needed for machine learning, as they are not available as
defaults in the Studio kernel.

.. code:: ipython3

    # Install smjsindustry SDK
    sdk_bucket = f's3://{notebook_artifact_bucket}/{notebook_sdk_prefix}'
    !aws s3 sync $sdk_bucket ./
    
    !pip install --no-index smjsindustry-1.0.0-py3-none-any.whl

.. code:: ipython3

    # import some packages
    import boto3
    import pandas as pd
    import sagemaker
    import smjsindustry

   **Note**: Step 1 and Step 2 will show you how to preprocess the
   training data and how to add MD&A Text features and NLP scores. You
   can also opt to use our provided preprocessed data
   ``sample_train_nlp_scores.csv`` and ``sample_test_nlp_scores.csv``
   skip Step 1&2 and directly go to Step 3.

Step 1: Prepare a Dataset
-------------------------

Here, we read in the dataframe curated by the SEC Retriever that is
already prepared as an example. The use of the Retriever is described in
another notebook provided, ``SEC_Retrieval_Summarizer_Scoring.ipynb``.
The industry codes shown here correspond to those in the `NAICS
system <https://www.census.gov/naics/>`__. We also attached the industry
codes from `Standard Industrial Classification (SIC)
Manual <https://www.osha.gov/data/sic-manual>`__.

Because 10-K/Q firms are filed once a quarter, each firm shows up
several instances in the dataset. When separating the dataset into train
and test sets, we made sure that firms only appear in either the train
or the test dataset, not in both. This ensures that the models are not
able to use the name of a firm from the training dataset to recognize
and classify firms in the test dataset.

The classification task here appears trivial but it is not; the MD&A
section of the forms includes very long texts. In a separate analysis,
we count the number of tokens (words) in each MD&A section for 12,144
filings, and obtain a mean of 5,307 tokens (sd=3,598 and interquartile
range of 3140 to 6505). Transformer models, such as BERT, usually handle
maximum sequence lengths of 512 or 1024 tokens. Therefore, it is
unlikely that this classification task will benefit from recent advances
in transformer models.

   **Important**: This example notebook uses data obtained from the SEC
   EDGAR database. You are responsible for complying with EDGAR’s access
   terms and conditions located in the `Accessing EDGAR
   Data <https://www.sec.gov/os/accessing-edgar-data>`__ page.

Process the raw data
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    # READ IN THE DATASETS (The file sizes are large. They are about 1 GB in total)
    train_df = pd.read_csv('sec_ind_train.csv')
    test_df = pd.read_csv('sec_ind_test.csv')

.. code:: ipython3

    # Remove the very small classes to simplify, if needed
    train_df = train_df[train_df.industry_code!="C"]
    train_df = train_df[train_df.industry_code!="F"]
    test_df = test_df[test_df.industry_code!="C"]
    test_df = test_df[test_df.industry_code!="F"]

You can find in the following cells that there are over 11,000 for the
train dataset and over 3,000 for the test dataset. Note that there’s a
label (class) imbalance underlying in the dataset.

.. code:: ipython3

    # Show classes
    print(train_df.shape, test_df.shape)
    train_df.groupby('industry_code').count()

.. code:: ipython3

    test_df.groupby('industry_code').count()

For demonstration purposes, take a sample from the original dataset to
reduce the time for training.

.. code:: ipython3

    sample_train_df = train_df.groupby('industry_code', group_keys=False).apply(pd.DataFrame.sample, n=80, random_state=12)

.. code:: ipython3

    sample_train_df.groupby('industry_code').count()

.. code:: ipython3

    sample_test_df = test_df.groupby('industry_code', group_keys=False).apply(pd.DataFrame.sample, n=20, random_state=12)

.. code:: ipython3

    sample_test_df.groupby('industry_code').count()

.. code:: ipython3

    # Save the smaller datasets for use
    sample_train_df.to_csv('sample_train.csv',index=False)
    sample_test_df.to_csv('sample_test.csv',index=False)

Step 2: Add NLP scores to the MD&A Text Features
------------------------------------------------

Here we use the NLP scoring API to add three additional numerical
features to the dataframe for a better classification performance. The
columns will carry scores of the various attributes of the text.

NLP scoring delivers a score as the fraction of words in a document that
are in one of the word lists. You can provide your own word list to
calculate the NLP scores, such as negative, positive, risk, uncertainty,
certainty, litigious, fraud and safe word lists.

The approach taken here does not use human-curated word lists such as
the popular dictionary from `Loughran and
McDonald <https://sraf.nd.edu/textual-analysis/resources/>`__, widely
used in academia. Instead, the word lists here are generated from word
embeddings trained on standard large text corpora where each word list
comprises words that are close to the concept word (e.g. “risk”) in
embedding space. These word lists may contain words that a human may
list out, but may still occur in the context of the concept word.

You can also calculate your own scoring type by specifying a new word
list.

**Technical notes**:

1. The data loader accesses a container to process the request. There
   might be some latency when starting up the container, which accounts
   for a few initial minutes. The actual filings extraction occurs after
   this.
2. The data loader only supports processing jobs with only one instance
   at the moment.
3. Users are not charged for the waiting time used when the instance is
   initializing (this takes 3-5 minutes).
4. The name of the processing job is shown in the run time log.
5. You can also access the processing job from the `SageMaker
   console <https://console.aws.amazon.com/sagemaker>`__. On the left
   navigation pane, choose Processing, Processing job.

Prepare a SageMaker session S3 bucket and folder to store processed data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sagemaker
    session = sagemaker.Session()
    bucket = session.default_bucket()
    mnist_folder='jumpstart_industry_mnist'

Construct a SageMaker processor for NLP scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    # CODE TO CALL THE SMJSINDUSTRY CONTAINER TO ADD NLP SCORE COLUMNS to test_df
    import smjsindustry
    from smjsindustry import NLPScoreType
    from smjsindustry import NLPScorer
    from smjsindustry import NLPScorerConfig
    
    score_types = [NLPScoreType.POSITIVE, NLPScoreType.NEGATIVE, NLPScoreType.SAFE]
    
    score_type_list = list(
        NLPScoreType(score_type, [])
        for score_type in score_types
    )
    
    nlp_scorer_config = NLPScorerConfig(score_type_list)
    
    nlp_score_processor = NLPScorer(
            sagemaker.get_execution_role(),         # loading job execution role
            1,                                      # number of ec2 instances to run the loading job, can support multiple instances
            'ml.c5.18xlarge',                       # ec2 instance type to run the loading job
            volume_size_in_gb=30,                   # size in GB of the EBS volume to use
            volume_kms_key=None,                    # KMS key for the processing volume
            output_kms_key=None,                    # KMS key ID for processing job outputs
            max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
            sagemaker_session=sagemaker.Session(),  # session object
            tags=None)                              # a list of key-value pairs

Run the NLP-scoring processing job on the training set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The processing job runs on a ``ml.c5.18xlarge`` instance to reduce the
running time. If ``ml.c5.18xlarge`` is not available in your AWS Region,
change to a different CPU-based instance. If you encounter error
messages that you’ve exceeded your quota, contact AWS Support to request
a service limit increase for `SageMaker
resources <https://console.aws.amazon.com/support/home#/>`__ you want to
scale up.

.. code:: ipython3

    nlp_score_processor.calculate(
        nlp_scorer_config, 
        "MDNA",                                                                               # input column
        'sample_train.csv',                                                                   # input from s3 bucket
        's3://{}/{}/{}'.format(bucket, mnist_folder, 'output'),                               # output s3 prefix (both bucket and folder names are required)
        'sample_train_nlp_scores.csv'                                                         # output file name
    )

Examine the dataframe of the tabular-and-text (TabText) data.

Note that it has a column for MD&A text, a categorical column for
industry code, and three numerical columns (``POSITIVE``, ``NEGATIVE``,
and ``SAFE``). In the next step, you’ll use this multimodal dataset to
train a model of AWS Gluon, which can accommodate the multimodal data.

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(mnist_folder, 'output', 'sample_train_nlp_scores.csv'), 'sample_train_nlp_scores.csv')
    df = pd.read_csv('sample_train_nlp_scores.csv')
    df.head()

Run the NLP-scoring processing job on the test set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    nlp_score_processor.calculate(
        nlp_scorer_config, 
        "MDNA",                                                                               # input column
        'sample_test.csv',                                                                    # input from s3 bucket
        's3://{}/{}/{}'.format(bucket, mnist_folder, 'output'),                               # output s3 prefix (both bucket and folder names are required)
        'sample_test_nlp_scores.csv'                                                          # output file name
    )

Examine the dataframe of the TabText data.

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(mnist_folder, 'output', 'sample_test_nlp_scores.csv'), 'sample_test_nlp_scores.csv')
    df = pd.read_csv('sample_test_nlp_scores.csv')
    df.head()

Step 3: Train the AutoGluon Model for Classification on the TabText Data Consists of the MD&A Texts, Industry Codes, and the NLP scores
---------------------------------------------------------------------------------------------------------------------------------------

Prepare the SageMaker training environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We create ``lib`` folder and ``requirements.txt`` file to store
AutoGluon related dependencies. These dependencies will be installed in
the training containers. For more information, see `Use third-party
libraries <https://sagemaker.readthedocs.io/en/stable/frameworks/mxnet/using_mxnet.html#use-third-party-libraries>`__
in the *Amazon SageMaker Python SDK documentation*.

Download autogluon installation packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    autogluon_bucket = f"s3://{notebook_artifact_bucket}/{notebook_autogluon_prefix}"
    !aws s3 sync $autogluon_bucket ./

.. code:: ipython3

    !mkdir -p model-training/lib
    !tar -zxvf autogluon.tar.gz -C model-training/lib --strip-components=1 --no-same-owner

Save paths for dependency requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    !cd model-training/lib && ls > ../requirements.txt
    !cd model-training && sed -i -e 's#^#lib/#' requirements.txt

The steps for training the AutoGluon classification model:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Read in the extended TabText dataframes created in the previous code
   blocks.
2. Normalize the NLP scores, as this usually helps improve the ML model.
3. Upload the training and test dataset to the session bucket.
4. Train and evaluate the model in MXNet. See more details in the
   **train.py**.
5. Generate the leaderboard to examine all the different models for
   performance.

.. code:: ipython3

    %%time
    %pylab inline
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    # Read in the prepared data files
    sample_train_nlp_df = pd.read_csv("sample_train_nlp_scores.csv")
    sample_test_nlp_df = pd.read_csv("sample_test_nlp_scores.csv")
    
    # Normalize the NLP score columns
    nlp_scores_names = ['negative', 'positive', 'safe']
    for col in nlp_scores_names:
        x = array(sample_train_nlp_df[col]).reshape(-1,1)
        sample_train_nlp_df[col] = scaler.fit_transform(x)
        x = array(sample_test_nlp_df[col]).reshape(-1,1)
        sample_test_nlp_df[col] = scaler.fit_transform(x)    

.. code:: ipython3

    import sagemaker
    session = sagemaker.Session()
    bucket = session.default_bucket()
    
    sample_train_nlp_df.to_csv("train_data.csv", index=False)
    sample_test_nlp_df.to_csv("test_data.csv", index=False)
    
    mnist_folder='jumpstart_mnist'
    train_s3_path = session.upload_data('train_data.csv', bucket=bucket, key_prefix=mnist_folder+'/'+'data')
    test_s3_path = session.upload_data('test_data.csv', bucket=bucket, key_prefix=mnist_folder+'/'+'data')

Run a SageMaker training job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The training job takes around 10 minutes with the sample dataset. If you
want to train a model with your own data, you may need to update the
training script ``train.py`` in the\ ``model-training`` folder. If you
want to use a GPU instance to achieve a better accuracy, please replace
``train_instance_type`` with the desired GPU instance and uncomment
``fit_args`` and ``hyperparameters`` to pass in the related arguments to
the training script as hyperparameters.

.. code:: ipython3

    from sagemaker.mxnet import MXNet
    
    # Define required label and additional parameters for Autogluon TabularPredictor
    init_args = {
      'label': 'industry_code'
    }
    
    # Define parameters for Autogluon TabularPredictor fit method
    #fit_args = {
    #  'ag_args_fit': {'num_gpus': 1}
    #}
    
    hyperparameters = {'init_args': str(init_args)}
    #hyperparameters = {'init_args': str(init_args), 'fit_args': str(fit_args)}
    
    tags = [{'Key' : 'AlgorithmName', 'Value' : 'AutoGluon-Tabular'}, 
            {'Key' : 'ProjectName', 'Value' : 'Jumpstart-gecko'},]
    
    estimator = MXNet(
        entry_point="train.py",
        role=sagemaker.get_execution_role(),
        train_instance_count=1,
        train_instance_type="ml.c5.2xlarge",
        framework_version="1.8.0",
        py_version="py37",
        source_dir="model-training",
        base_job_name='jumpstart-example-gecko-mnist',
        hyperparameters=hyperparameters,
        tags=tags,
        disable_profiler=True,
        debugger_hook_config=False,
        enable_network_isolation=True,  # Set enable_network_isolation=True to ensure a security running environment
    )
    
    inputs = {'training': train_s3_path, 'testing': test_s3_path}
    
    estimator.fit(inputs)

Download Model Outputs
^^^^^^^^^^^^^^^^^^^^^^

We download the following files (training job artifacts) from the
SageMaker session’s default S3 bucket: \* ``leaderboard.csv`` \*
``predictions.csv`` \* ``feature_importance.csv`` \* ``evaluation.json``

.. code:: ipython3

    import boto3 
    
    s3_client = boto3.client("s3")
    job_name = estimator._current_job_name
    s3_client.download_file(bucket, f"{job_name}/output/output.tar.gz", "output.tar.gz")
    !tar -xvzf output.tar.gz

Score details of each model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    leaderboard = pd.read_csv("leaderboard.csv")
    leaderboard

The result of the training evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    import json
    
    with open('evaluation.json') as f:
        data = json.load(f)
    print(data)

Classification report and Confusion matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import networkx as nx
    import seaborn as sns
    
    y_true = sample_test_nlp_df[init_args['label']]
    y_pred = pd.read_csv("predictions.csv")['industry_code']
    
    #Classification report
    report_dict = classification_report(
            y_true, y_pred, output_dict=True, labels=['B','D','E','G','H','I']
            )
    report_dict.pop('accuracy', None)
    report_dict_df = pd.DataFrame(report_dict).T
    print(report_dict_df)
    report_dict_df.to_csv("classification_report.csv", index=True)
    
    #Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['B','D','E','G','H','I'])
    cm_df = pd.DataFrame(cm, ['B','D','E','G','H','I'], ['B','D','E','G','H','I'])
    sns.set(font_scale=1)
    cmap = "coolwarm"
    sns.heatmap(cm_df, annot=True, fmt="d", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.show()
    plt.savefig("confusion_matrix.png")

Summary
-------

1. We curated a TabText dataframe concatenating text, tabular, and
   categorical data.
2. We demonstrated how to do ML on a TabText (multimodal) data using
   `AutoGluon <https://github.com/awslabs/autogluon>`__.

Clean Up
--------

After you are done using this notebook, delete the model artifacts and
other resources to avoid any incurring charges.

   **Caution:** You need to manually delete resources that you may have
   created while running the notebook, such as Amazon S3 buckets for
   model artifacts, training datasets, processing artifacts, and Amazon
   CloudWatch log groups.

For more information about cleaning up resources, see `Clean
Up <https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html>`__
in the *Amazon SageMaker Developer Guide*.

Licence
-------

The SageMaker JumpStart Industry product and its related materials are
under the `Legal License
Terms <https://jumpstart-cache-alpha-us-west-2.s3.us-west-2.amazonaws.com/smfinance-notebook-dependency/legal_file.txt>`__.

