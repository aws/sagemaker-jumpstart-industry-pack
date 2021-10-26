.. important::

   This page is for preview purposes only to show the content of
   `Amazon SageMaker JumpStart Industry Example Notebooks
   <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart-industry.html#studio-jumpstart-industry-examples>`_.

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

Simple Construction of a Multimodal Dataset from SEC Filings and NLP Scores
===========================================================================

Amazon SageMaker is a fully managed service that provides developers and
data scientists with the ability to build, train, and deploy machine
learning (ML) models quickly. Amazon SageMaker removes the heavy lifting
from each step of the machine learning process to make it easier to
develop high-quality models. The SageMaker Python SDK makes it easy to
train and deploy models in Amazon SageMaker with several different
machine learning and deep learning frameworks, including PyTorch and
TensorFlow.

This notebook shows how to use Amazon SageMaker to deploy a simple
solution to retrieve U.S. Securities and Exchange Commission (SEC)
filings and construct a dataframe of mixed tabular and text data, called
TabText. This is a first step in multimodal machine learning.

   **Important**: This example notebook is for demonstrative purposes
   only. It is not financial advice and should not be relied on as
   financial or investment advice.

Why SEC Filings?
----------------

Financial NLP is a subset of the rapidly increasing use of ML in
finance, but it is the largest; for more information, see this `survey
paper <https://arxiv.org/abs/2002.05786>`__. The starting point for a
vast amount of financial natural language processing (NLP) is text in
SEC filings. The SEC requires companies to report different types of
information related to various events involving companies. To find the
full list of SEC forms, see `Forms List <https://www.sec.gov/forms>`__
in the *Securities and Exchange Commission (SEC) website*.

SEC filings are widely used by financial services companies as a source
of information about companies. Financial services companies may use
this information as part of trading, lending, investment, and risk
management decisions. Because these filings are required, they are of
high quality. They contain forward-looking information that helps with
forecasts and are written with a view to the future. In addition, in
recent times, the value of historical time-series data has degraded,
because economies have been structurally transformed by trade wars,
pandemics, and political upheavals. Therefore, text as a source of
forward-looking information has been increasing in relevance.

There has been an exponential growth in downloads of SEC filings.To find
out more, see `“How to Talk When a Machine is Listening: Corporate
Disclosure in the Age of AI” <https://www.nber.org/papers/w27950>`__.
This paper reports that the number of machine downloads of corporate
10-K and 10-Q filings increased from 360,861 in 2003 to 165,318,719 in
2016.

There is a vast body of academic and practitioner research that is based
on financial text, a significant portion of which is based on SEC
filings. A recent review article summarizing this work is `“Textual
Analysis in Finance
(2020)” <https://www.annualreviews.org/doi/abs/10.1146/annurev-financial-012820-032249>`__.

What Does SageMaker Do?
-----------------------

SEC filings are downloaded from the `SEC’s Electronic Data Gathering,
Analysis, and Retrieval (EDGAR)
website <https://www.sec.gov/edgar/search-and-access>`__, which provides
open data access. EDGAR is the primary system under the SEC for
companies and others submitting documents under the Securities Act of
1933, the Securities Exchange Act of 1934, the Trust Indenture Act of
1939, and the Investment Company Act of 1940. EDGAR contains millions of
company and individual filings. The system processes about 3,000 filings
per day, serves up 3,000 terabytes of data to the public annually, and
accommodates 40,000 new filers per year on average.

There are several ways to download the data, and some open source
packages available to extract the text from these filings. However,
these require extensive programming and are not always easy to use.
Following, you can find a simple *one*-API call that will create a
dataset in a few lines of code, for any period of time and for a large
number of tickers.

This SageMaker JumpStart Industry example notebook wraps the extraction
functionality into a SageMaker processing container. This notebook also
provides code samples that enable users to download a dataset of filings
with meta data, such as dates and parsed plain text that can then be
used for machine learning using other SageMaker tools. You only need to
specify a date range and a list of ticker symbols (or Central Index Key
codes (CIK) codes, which are the SEC assigned identifier). This
notebooks does the rest.

Currently, this solution supports extracting a popular subset of SEC
forms in plain text (excluding tables). These are 10-K, 10-Q, 8-K, 497,
497K, S-3ASR and N-1A. For each of these, you can find examples
following and a brief description of each form. For the 10-K and 10-Q
forms, filed every year or quarter, the solution also extracts the
Management Discussion and Analysis (MD&A) section, which is the primary
forward-looking section in the filing. This section is the one most
widely used in financial text analysis. This information is provided
automatically in a separate column of the dataframe alongside the full
text of the filing.

The extracted dataframe is written to Amazon S3 storage and to the local
notebook instance.

Security Requirements
---------------------

We provide a client library named SageMaker JumpStart Industry Python
SDK (``smjsindustry``). The library provides the capability of running
processing containers in customers’ Amazon virtual private cloud (VPC).
More specifically, when calling ``smjsindustry`` API operations,
customers can specify their VPC configurations such as ``subnet-id`` and
``security-group-id``. SageMaker will launch ``smjsindustry`` processing
containers in the VPC implied by the subnets. The intercontainer traffic
is specified by the security groups.

Customers can also secure data at rest using their own AWS KMS keys. The
``smjsindustry`` package encrypts EBS volumes and S3 data if users
passes the AWS KMS keys information to the ``volume_kms_key`` and
``output_kms_key`` arguments of a SageMaker processor.

SageMaker Studio Kernel Setup
-----------------------------

Recommended kernel is **Python 3 (Data Science)**. *DO NOT* use the
**Python 3 (SageMaker JumpStart Data Science 1.0)** kernel because there
are some differences in preinstalled dependency. For the instance type,
using a larger instance with sufficient memory can be helpful to
download the following materials.

Load Data, SDK, and Dependencies
--------------------------------

The following code cells download the ```smjsindustry``
SDK <https://pypi.org/project/smjsindustry/>`__, dependencies, and
dataset from an S3 bucket prepared by SageMaker JumpStart Industry. You
will learn how to use the ``smjsindustry`` SDK which contains various
APIs to curate SEC datasets. The dataset in this example was
synthetically generated using the ``smjsindustry`` package’s SEC Forms
Retrieval tool. For more information, see the `SageMaker JumpStart
Industry Python SDK
documentation <https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/notebooks/index.html>`__.

Install the ``smjsindustry`` library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We deliver APIs through the ``smjsindustry`` client library. The first
step requires pip installing a Python package that interacts with a
SageMaker processing container. The retrieval, parsing, transforming,
and scoring of text is a complex process and uses many different
algorithms and packages. To make this seamless and stable for the user,
the functionality is packaged into an S3 bucket. For installation and
maintenance of the workflow, this approach reduces your eﬀort to a pip
install followed by a single API call.

The following code blocks copy the wheel file to install the
``smjsindustry`` library. It also downloads a synthetic dataset and
dependencies to demonstrate the functionality of curating the TabText
dataframe.

.. code:: ipython3

    notebook_artifact_bucket = 'jumpstart-cache-alpha-us-west-2'
    notebook_data_prefix = 'smfinance-notebook-data/smjsindustry-tutorial'
    notebook_sdk_prefix = 'smfinance-notebook-dependency/smjsindustry'

.. code:: ipython3

    # Download example dataset
    data_bucket = f's3://{notebook_artifact_bucket}/{notebook_data_prefix}'
    ! aws s3 sync $data_bucket ./

Install the ``smjsindustry`` library and dependencies by running the
following code block; the packages are needed for machine learning but
aren’t available as defaults in SageMaker Studio.

.. code:: ipython3

    # Install smjsindustry SDK
    sdk_bucket = f's3://{notebook_artifact_bucket}/{notebook_sdk_prefix}'
    !aws s3 sync $sdk_bucket ./

    !pip install --no-index smjsindustry-1.0.0-py3-none-any.whl

.. code:: ipython3

    %pylab inline

The preceding line loads in several standard packages, including NumPy,
SciPy, and matplotlib.

Next, we import required packages and load the S3 bucket from SageMaker
session, as shown below.

.. code:: ipython3

    import boto3
    import pandas as pd
    import sagemaker
    import smjsindustry

.. code:: ipython3

    from smjsindustry.finance import utils
    from smjsindustry import NLPScoreType, NLPSCORE_NO_WORD_LIST
    from smjsindustry import NLPScorerConfig, JaccardSummarizerConfig, KMedoidsSummarizerConfig
    from smjsindustry import Summarizer, NLPScorer
    from smjsindustry.finance.processor import DataLoader, SECXMLFilingParser
    from smjsindustry.finance.processor_config import EDGARDataSetConfig

.. code:: ipython3

    # Prepare the SageMaker session's default S3 bucket and a folder to store processed data
    session = sagemaker.Session()
    bucket = session.default_bucket()
    sec_processed_folder='jumpstart_industry_sec_processed'

Next, you can find examples of how to extract the diﬀerent forms.

SEC Filing Retrieval
--------------------

SEC Forms 10-K/10-Q
~~~~~~~~~~~~~~~~~~~

10-K/10-Q forms are quarterly reports required to be filed by companies.
They contain full disclosure of business conditions for the company and
also require forward-looking statements of future prospects, usually
written into a section known as the “Management Discussion & Analysis”
section. There also can be a section called “Forward-Looking
Statements”. For more information, see `Form
10-K <https://www.investor.gov/introduction-investing/investing-basics/glossary/form-10-k>`__
in the *Investor.gov webpage*.

Each year firms file three 10-Q forms (quarterly reports) and one 10-K
(annual report). Thus, there are in total four reports each year. The
structure of the forms is displayed in a table of contents.

The SEC filing retrieval supports the downloading and parsing of 10-K,
10-Q, 8-K, 497, 497K, S-3ASR and N-1A, seven form types for the tickers
or CIKs specified by the user. The following block of code will download
full text of the forms and convert it into a dataframe format using a
SageMaker session. The code is self-explanatory, and offers customized
options to the users.

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

Users may update any of the settings in the ``data_loader`` section of
the code block below, and in the ``dataset_config`` section. For a very
long list of tickers or CIKs, the job will run for a while, and the
``...`` stream will indicate activity as it proceeds.

| **NOTE**: We recommend that you use CIKs as the input. The tickers are
  internally converted to CIKs according to this `mapping
  file <https://www.sec.gov/include/ticker.txt>`__.
| One ticker can map to multiple CIKs, but this solution supports only
  the latest ticker to CIK mapping. Make sure to provide the old CIKs in
  the input when you want historical filings.

The following code block shows how to use the SEC Retriever API. You
specify system resources (or just choose the defaults below). Also
specify the tickers needed, the SEC forms needed, the date range, and
the location and name of the file in S3 where the curated data file will
be stored in CSV format. The output will shows the runtime log from the
SageMaker processing container and indicates when it is completed.

   **Important**: This example notebook uses data obtained from the SEC
   EDGAR database. You are responsible for complying with EDGAR’s access
   terms and conditions located in the `Accessing EDGAR
   Data <https://www.sec.gov/os/accessing-edgar-data>`__ page.

.. code:: ipython3

    %%time

    dataset_config = EDGARDataSetConfig(
        tickers_or_ciks=['amzn','goog', '27904', 'FB'],  # list of stock tickers or CIKs
        form_types=['10-K', '10-Q'],                     # list of SEC form types
        filing_date_start='2019-01-01',                  # starting filing date
        filing_date_end='2020-12-31',                    # ending filing date
        email_as_user_agent='test-user@test.com')        # user agent email

    data_loader = DataLoader(
        role=sagemaker.get_execution_role(),    # loading job execution role
        instance_count=1,                       # instances number, limit varies with instance type
        instance_type='ml.c5.2xlarge',          # instance type
        volume_size_in_gb=30,                   # size in GB of the EBS volume to use
        volume_kms_key=None,                    # KMS key for the processing volume
        output_kms_key=None,                    # KMS key ID for processing job outputs
        max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
        sagemaker_session=sagemaker.Session(),  # session object
        tags=None)                              # a list of key-value pairs

    data_loader.load(
        dataset_config,
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),     # output s3 prefix (both bucket and folder names are required)
        'dataset_10k_10q.csv',                                              # output file name
        wait=True,
        logs=True)

Output
^^^^^^

The output of the DataLoader processing job is a dataframe. This job
includes 32 filings (4 companies for 8 quarters). The CSV file is
downloaded from S3 and then read into a dataframe, as shown in the
following few code blocks.

The filing date comes within a month of the end date of the reporting
period. The filing date is displayed in the dataframe. The column
``"text"`` contains the full plain text of the filing but the tables are
not extracted. The values in the tables in the filings are balance-sheet
and income-statement data (numeric and tabular) and are easily available
elsewhere as they are reported in numeric databases. The last column
(``"mdna"``) of the dataframe comprises the Management Discussion &
Analysis section, which is also included in the ``"text"`` column.

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'dataset_10k_10q.csv'), 'dataset_10k_10q.csv')
    data_frame_10k_10q = pd.read_csv('dataset_10k_10q.csv')
    data_frame_10k_10q

As an example of a clean parse, print out the text of the first filing.

.. code:: ipython3

    print(data_frame_10k_10q.text[0])

To read the MD&A section, use the following code to print out the
section for the second filing in the dataframe.

.. code:: ipython3

    print(data_frame_10k_10q.mdna[1])

SEC Form 8-K
~~~~~~~~~~~~

This form is filed for material changes in business conditions. This
`Form 8-K
page <https://www.sec.gov/fast-answers/answersform8khtm.html>`__
describes the form requirements and various conditions for publishing a
8-K filing. Because there is no set cadence to these filings, several
8-K forms might be filed within a year, depending on how often a company
experiences material changes in business conditions.

The API call below is the same as for the 10-K forms; simply change the
form type ``8-K`` to ``10-K``.

.. code:: ipython3

    %%time

    dataset_config = EDGARDataSetConfig(
        tickers_or_ciks=['amzn','goog', '27904', 'FB'],  # list of stock tickers or CIKs
        form_types=['8-K'],                              # list of SEC form types
        filing_date_start='2019-01-01',                  # starting filing date
        filing_date_end='2020-12-31',                    # ending filing date
        email_as_user_agent='test-user@test.com')        # user agent email

    data_loader = DataLoader(
        role=sagemaker.get_execution_role(),    # loading job execution role
        instance_count=1,                       # instances number, limit varies with instance type
        instance_type='ml.c5.2xlarge',          # instance type
        volume_size_in_gb=30,                   # size in GB of the EBS volume to use
        volume_kms_key=None,                    # KMS key for the processing volume
        output_kms_key=None,                    # KMS key ID for processing job outputs
        max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
        sagemaker_session=sagemaker.Session(),  # session object
        tags=None)                              # a list of key-value pairs

    data_loader.load(
        dataset_config,
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),    # output s3 prefix (both bucket and folder names are required)
        'dataset_8k.csv',                                                  # output file name
        wait=True,
        logs=True)

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'dataset_8k.csv'), 'dataset_8k.csv')
    data_frame_8k = pd.read_csv('dataset_8k.csv')
    data_frame_8k

As noted, 8-K forms do not have a fixed cadence, and they depend on the
number of times a company changes the material. Therefore, the number of
forms varies over time.

Next, print the plain text of the first 8-K form in the dataframe.

.. code:: ipython3

    print(data_frame_8k.text[0])

Other SEC Forms
~~~~~~~~~~~~~~~

We also support SEC forms 497, 497K, S-3ASR, N-1A, 485BXT, 485BPOS,
485APOS, S-3, S-3/A, DEF 14A, SC 13D and SC 13D/A.

SEC Form 497
^^^^^^^^^^^^

Mutual funds are required to file Form 497 to disclose any information
that is material for investors. Funds file their prospectuses using this
form as well as proxy statements. The form is also used for Statements
of Additional Information (SAI). The forward-looking information in Form
497 comprises the detailed company history, financial statements, a
description of products and services, an annual review of the
organization, its operations, and the markets in which the company
operates. Much of this data is usually audited so is of high quality.
For more information, see `SEC Form
497 <https://www.investopedia.com/terms/s/sec-form-497.asp>`__.

SEC Form 497K
^^^^^^^^^^^^^

This is a summary prospectus. It describes the fees and expenses of the
fund, its principal investment strategies, principal risks, past
performance if any, and some administrative information. Many such forms
are filed for example, in Q4 of 2020 a total of 5,848 forms of type 497K
were filed.

SEC Form S-3ASR
^^^^^^^^^^^^^^^

The S-3ASR is an automatic shelf registration statement which is
immediately effective upon filing for use by well-known seasoned issuers
to register unspecified amounts of different specified types of
securities. This Registration Statement is for the registration of
securities under the Securities Act of 1933.

SEC Form N-1A
^^^^^^^^^^^^^

This registration form is required for establishing open-end management
companies. The form can be used for registering both open-end mutual
funds and open-end exchange traded funds (ETFs). For more information,
see `SEC Form
N-1A <https://www.investopedia.com/terms/s/sec-form-n-1a.asp>`__.

.. code:: ipython3

    %%time

    dataset_config = EDGARDataSetConfig(
        tickers_or_ciks=['zm', '709364', '1829774'],   # list of stock tickers or CIKs, 709364 is the CIK for ROYCE FUND and 1829774 is the CIK for James Alpha Funds Trust
        form_types=['497', '497K', 'S-3ASR', 'N-1A'],  # list of SEC form types
        filing_date_start='2021-01-01',                # starting filing date
        filing_date_end='2021-02-01',                  # ending filing date
        email_as_user_agent='test-user@test.com')      # user agent email

    data_loader = DataLoader(
        role=sagemaker.get_execution_role(),         # loading job execution role
        instance_count=1,                            # instances number, limit varies with instance type
        instance_type='ml.c5.2xlarge',               # instance type
        volume_size_in_gb=30,                        # size in GB of the EBS volume to use
        volume_kms_key=None,                         # KMS key for the processing volume
        output_kms_key=None,                         # KMS key ID for processing job outputs
        max_runtime_in_seconds=None,                 # timeout in seconds. Default is 24 hours.
        sagemaker_session=sagemaker.Session(),       # session object
        tags=None)                                   # a list of key-value pairs

    data_loader.load(
        dataset_config,
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),    # output s3 prefix (both bucket and folder names are required)
        'dataset_other_forms.csv',                                         # output file name
        wait=True,
        logs=True)

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'dataset_other_forms.csv'), 'dataset_other_forms.csv')
    data_frame_other_forms = pd.read_csv('dataset_other_forms.csv')
    data_frame_other_forms

.. code:: ipython3

    # Example of 497 form
    print(data_frame_other_forms.text[2])

.. code:: ipython3

    # Example of 497K form
    print(data_frame_other_forms.text[4])

.. code:: ipython3

    # Example of S-3ASR form
    print(data_frame_other_forms.text[0])

.. code:: ipython3

    # Example of N-1A form
    print(data_frame_other_forms.text[1])

SEC Filing Parser
-----------------

If you have the SEC filings ready locally or in an S3 bucket, you can
use the SEC Filing Parser API to parse the raw file and to generate
clear and structured text.

.. code:: ipython3

    %%time
    parser = SECXMLFilingParser(
        role=sagemaker.get_execution_role(),         # loading job execution role
        instance_count=1,                            # instances number, limit varies with instance type
        instance_type='ml.c5.2xlarge',               # instance type
        sagemaker_session=sagemaker.Session()        # Session object
    )
    parser.parse(
        'xml',                                                              # local input folder or S3 path
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),     # output s3 prefix (both bucket and folder names are required)
    )

.. code:: ipython3

    xml_file_name = ['0001018724-21-000002.txt', '0001018724-21-000004.txt']
    parsed_file_name = ["parsed-"+ name for name in xml_file_name]

    client = boto3.client('s3')
    for file in parsed_file_name:
        client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', file), file)

    parsed_res = open(parsed_file_name[0], "r")
    print(parsed_res.read())

SEC Filing Summarizer
---------------------

The ``smjsindustry`` Python SDK provides two text summarizers that
extracts concise summaries while preserving key information and overall
meaning. ``JaccardSummarizer`` and ``KMedoidsSummarizer`` are the text
summarizers adopted to the ``smjsindustry`` Python SDK.

You can configure a ``JaccardSummarizer`` processor or a
``KMedoidsSummarizer`` processor using the ``smjsindustry`` library, and
run a processing job using the SageMaker Python SDK. To achieve better
performance and reduced training time, the processing job can be
initiated with multiple instances.

**Technical Notes**:

1. The summarizers send SageMaker processing job requests to processing
   containers. It might take a few minutes when spinning up a processing
   container. The actual filings extraction start after the initial
   spin-up.
2. You are not charged for the waiting time used for the initial
   spin-up.
3. You can run processing jobs in multiple instances.
4. The name of the processing job is shown in the runtime log.
5. You can also access the processing job from the `SageMaker
   console <https://console.aws.amazon.com/sagemaker>`__. On the left
   navigation pane, choose Processing, Processing job.
6. VPC mode is supported for the summarizers.

Jaccard Summarizer
~~~~~~~~~~~~~~~~~~

The Jaccard summarizer uses the `Jaccard
index <https://en.wikipedia.org/wiki/Jaccard_index>`__. It provides the
main theme of a document by extracting the sentences with the greatest
similarity among all sentences. The metric calculates the number of
common words between two sentences normalized by the size of the
superset of the words in the two sentences.

You can use the ``summary_size``, ``summary_percentage``,
``max_tokens``, and ``cutoff`` parameters to limit the size of the docs
to be summarized (see **Example 1**).

You can also provide your own vocabulary to calculate Jaccard
similarities between sentences (see **Example 2**).

The Jaccard summarizer is an extractive summarizer (not abstractive).
There are two main reasons for adopting this extractive summarizer: -
One, the extractive approach retains the original sentences and thus
preserves the legal meaning of the sentences. - Two, it works fast on
very long text as we have in SEC filings. Long text is not easily
handled by abstractive summarizers that are based on embeddings from
transformers that can ingest a limited number of words.

**Two examples** are shown below: - In **Example 1**, JaccardSummarizer
for the ``'dataset_10k_10q.csv'`` data (created by data loader) runs
against the ``'text'`` column, resulting in a summary of 10% of the
original text length. - In **Example 2**, JaccardSummarizer for the
``'dataset_10k_10q.csv'`` data (created by data loader) runs against the
``'text'`` column. The summarizer uses the ``custom_vocabulary`` list
set, which is the union of the customized positive and negative word
lists. This creates summary of sentences containing more positive and
negative connotations.

Example 1
^^^^^^^^^

.. code:: ipython3

    %%time
    jaccard_summarizer_config = JaccardSummarizerConfig(summary_percentage = 0.1)

    jaccard_summarizer = Summarizer(
                    role = sagemaker.get_execution_role(),                # loading job execution role
                    instance_count=1,                                     # instances number, limit varies with instance type
                    instance_type='ml.c5.2xlarge',                        # instance type
                    sagemaker_session=sagemaker.Session())                # Session object

    jaccard_summarizer.summarize(
        jaccard_summarizer_config,
        'text',                                                             # text column name
        './dataset_10k_10q.csv',                                            # input file path
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),     # output s3 prefix (both bucket and folder names are required)
        'Jaccard_Summaries.csv',                                            # output file name
        new_summary_column_name="summary")                                  # add column "summary"

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'Jaccard_Summaries.csv'), 'Jaccard_Summaries.csv')
    Jaccard_summaries = pd.read_csv('Jaccard_Summaries.csv')
    Jaccard_summaries.head()

Example 2
^^^^^^^^^

Here is the second example, focusing on summaries with sentences
containing more positive and negative words.

.. code:: ipython3

    %%time

    positive_word_list = pd.read_csv('positive_words.csv')
    negative_word_list = pd.read_csv('negative_words.csv')
    custom_vocabulary = set(list(positive_word_list) + list(negative_word_list))

    jaccard_summarizer_config = JaccardSummarizerConfig(summary_percentage = 0.1, vocabulary = custom_vocabulary)

    jaccard_summarizer = Summarizer(
                    role = sagemaker.get_execution_role(),                # loading job execution role
                    instance_count=1,                                     # instances number, limit varies with instance type
                    instance_type='ml.c5.2xlarge',                        # instance type
                    sagemaker_session=sagemaker.Session())                # Session object

    jaccard_summarizer.summarize(
        jaccard_summarizer_config,
        'text',                                                             # text column name
        './dataset_10k_10q.csv',                                            # input file path
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),     # output s3 prefix (both bucket and folder names are required)
        'Jaccard_Summaries_pos_neg.csv',                                    # output file name
        new_summary_column_name="summary")                                  # add column "summary"

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'Jaccard_Summaries_pos_neg.csv'), 'Jaccard_Summaries_pos_neg.csv')
    Jaccard_summaries = pd.read_csv('Jaccard_Summaries_pos_neg.csv')
    Jaccard_summaries.head()

KMedoids Summarizer
~~~~~~~~~~~~~~~~~~~

The k-medoids summarizer clusters sentences and produces the medoid of
each cluster as summary. You can caculate the distance for clustering by
choosing one of the following distance metrics: ``'euclidean'``,
``'cosine'``, or ``'dot-product'``. Medoid initialization methods
include ``'random'``, ``'heuristic'``, ``'k-medoids++'``, and
``'build'``. You need to enter these options to the k-medoids summarizer
configuration (``KMedoidsSummarizerConfig``) in the first line of the
following code block. Available options are: - For ``metric``,
``{'euclidean', 'cosine', 'dot-product'}`` - For ``init``,
``{'random', 'heuristic', 'k-medoids++', 'build'}``

The size of the summary is specified as the number of sentences needed
in the summary.

**Two examples** are shown below: - In **Example 1**, KMedoidsSummarizer
for the ``'dataset_10k_10q.csv'`` data (created by data loader above)
runs against the ‘text’ column with only one instance. - In **Example
2**, KMedoidsSummarizer for the ``'dataset_8k.csv'`` data (created by
data loader above) runs against the ‘text’ column with two instances.

For the same reasons as stated for the Jaccard summarizer, the k-medoids
summarizer is also an extractive one.

Example 1
^^^^^^^^^

.. code:: ipython3

    %%time

    kmedoids_summarizer_config = KMedoidsSummarizerConfig(summary_size = 100)

    kmedoids_summarizer = Summarizer(
        sagemaker.get_execution_role(),         # loading job execution role
        instance_count = 1,                     # instances number, limit varies with instance type
        instance_type = 'ml.c5.2xlarge',        # instance type
        volume_size_in_gb=30,                   # size in GB of the EBS volume to use
        volume_kms_key=None,                    # KMS key for the processing volume
        output_kms_key=None,                    # KMS key ID for processing job outputs
        max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
        sagemaker_session = sagemaker.Session(),
        tags=None
    )

    kmedoids_summarizer.summarize(
        kmedoids_summarizer_config,
        "text",                                                                                           # text column name
        's3://{}/{}/{}/{}'.format(bucket, sec_processed_folder, 'output', 'dataset_10k_10q.csv'),         # input from s3 bucket
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),                                   # output s3 prefix (both bucket and folder names are required)
        'KMedoids_summaries.csv',                                                                         # output file name
        new_summary_column_name="summary",                                                                # add column "summary"
    )

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'KMedoids_summaries.csv'), 'KMedoids_summaries.csv')
    KMedoids_summaries = pd.read_csv('KMedoids_summaries.csv')
    KMedoids_summaries.head()

Example 2
^^^^^^^^^

.. code:: ipython3

    %%time

    kmedoids_summarizer_config = KMedoidsSummarizerConfig(summary_size = 100)

    kmedoids_summarizer = Summarizer(
        sagemaker.get_execution_role(),         # loading job execution role
        instance_count = 2,                     # instances number, limit varies with instance type
        instance_type = 'ml.c5.2xlarge',        # instance type
        volume_size_in_gb=30,                   # size in GB of the EBS volume to use
        volume_kms_key=None,                    # KMS key for the processing volume
        output_kms_key=None,                    # KMS key ID for processing job outputs
        max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
        sagemaker_session = sagemaker.Session(),
        tags=None
    )

    kmedoids_summarizer.summarize(
        kmedoids_summarizer_config,
        "text",                                                              # text column name
        "dataset_8k.csv",                                                    # input file path
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),      # output s3 prefix (both bucket and folder names are required)
        'KMedoids_summaries_multi_instance.csv',                             # output file name
        new_summary_column_name="summary",                                   # add column "summary"
    )


.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'KMedoids_summaries_multi_instance.csv'), 'KMedoids_summaries_multi_instance.csv')
    KMedoids_summaries_multi_instance = pd.read_csv('KMedoids_summaries_multi_instance.csv')
    KMedoids_summaries_multi_instance.head()

SEC Filing NLP Scoring
----------------------

The ``smjsindustry`` library provides 11 NLP score types by default:
``positive``, ``negative``, ``litigious``, ``polarity``, ``risk``,
``readability``, ``fraud``, ``safe``, ``certainty``, ``uncertainty``,
and ``sentiment``. Each score (except readability and sentiment) has its
word list, which is used for scanning and matching with an input text
dataset.

-  The ``readability`` score type is calculated adopting the `Gunning
   fog index <https://en.wikipedia.org/wiki/Gunning_fog_index>`__.
-  The ``sentiment`` score type adopts `VADER sentiment analysis
   method <https://pypi.org/project/vaderSentiment/>`__.
-  The ``polarity`` score type uses the ``positive`` and ``negative``
   word lists.
-  The rest of the NLP score types (``positive``, ``negative``,
   ``litigious``, ``risk``, ``fraud``, ``safe``, ``certainty``, and
   ``uncertainty``) evaluates the similarity (word frequency) with their
   corresponding word lists. For example, the ``positive`` NLP score has
   its own word list that contains “positive” meanings. To measure the
   ``positive`` score, the NLP scorer calculates the proportion of words
   out of the entire texts, by counting every readings of the words that
   are in the word list of the ``positive`` score. Before matching, the
   words are stemmed to match different tenses of the same word. You can
   provide your own word list to calculate the predefined NLP scores or
   create your own score with a new word list.

The NLP score types do not use human-curated word lists such as the
dictionary from `Loughran and
McDonald <https://sraf.nd.edu/textual-analysis/resources/>`__, which is
widely used in academia. Instead, the word lists are generated from word
embeddings trained on standard large text corpora; each word list
comprises words that are close to the concept word (such as
``positive``, ``negative``, and ``risk`` in this case) in an embedding
space. These word lists may contain words that a human might list out,
but might still occur in the context of the concept word.

These NLP scores are added as new numerical columns to the text
dataframe; this creates a multimodal dataframe, which is a mixture of
tabular data and longform text, called **TabText**. When submitting this
multimodal dataframe for ML, it is a good idea to normalize the columns
of NLP scores (usually with standard normalization or min-max scaling).

**Technical notes**:

1. The NLPScorer sends SageMaker processing job requests to processing
   containers. It might take a few minutes when spinning up a processing
   container. The actual filings extraction start after the initial
   spin-up.
2. You are not charged for the waiting time used for the initial
   spin-up.
3. You can run processing jobs in multiple instances.
4. The name of the processing job is shown in the runtime log.
5. You can also access the processing job from the `SageMaker
   console <https://console.aws.amazon.com/sagemaker>`__. On the left
   navigation pane, choose Processing, Processing job.
6. NLP scoring can be slow for massive documents such as SEC filings,
   which contain anywhere from 20K-100K words. Matching to word lists
   (usually ~200 words or more) can be time-consuming. This is why we
   have enabled automatic distribution of the rows of the dataframe for
   this task over multiple EC2 instances. In the example below, this is
   distributed over 4 instances and the run logs show the different
   instances in different colors. The user does not need to code up the
   distributed processing task here, it is done automatically when the
   number of instances is specified.
7. VPC mode is supported in this API.

**Three examples** are shown below: - In **Example 1**, 11 types of NLP
scores for the ``'dataset_10k_10q.csv'`` data (created by the
``data_loader``) is generated against the ``'text'`` column. - In
**Example 2**, customized positive and negative word lists are provided
to calculate the positive and negative NLP scores for the
``'dataset_10k_10q.csv'`` data (created by data loader above) against
the ``'text'`` column. - In **Example 3**, a customized score type, in
this case ``'societal'``, is created using a ``'societal'`` word list.
``'dataset_10k_10q.csv'`` data is loaded from a local file path.

The processing job runs on ``ml.c5.18xlarge`` to reduce the running
time. If ``ml.c5.18xlarge`` is not available in your AWS Region, change
to a different CPU-based instance. If you encounter error messages that
you’ve exceeded your quota, contact AWS Support to request a service
limit increase for `SageMaker
resources <https://console.aws.amazon.com/support/home#/>`__ you want to
scale up.

Example 1
^^^^^^^^^

It takes about 1 hour to run the following processing job because it
computes the entire 11 types of NLP scores.

.. code:: ipython3

    %%time

    import smjsindustry
    from smjsindustry import NLPScoreType, NLPSCORE_NO_WORD_LIST
    from smjsindustry import NLPScorer
    from smjsindustry import NLPScorerConfig

    score_type_list = list(
        NLPScoreType(score_type, [])
        for score_type in NLPScoreType.DEFAULT_SCORE_TYPES
        if score_type not in NLPSCORE_NO_WORD_LIST
    )
    score_type_list.extend([NLPScoreType(score_type, None) for score_type in NLPSCORE_NO_WORD_LIST])

    nlp_scorer_config = NLPScorerConfig(score_type_list)

    nlp_score_processor = NLPScorer(
            sagemaker.get_execution_role(),         # loading job execution role
            1,                                      # instances number, limit varies with instance type
            'ml.c5.18xlarge',                       # ec2 instance type to run the loading job
            volume_size_in_gb=30,                   # size in GB of the EBS volume to use
            volume_kms_key=None,                    # KMS key for the processing volume
            output_kms_key=None,                    # KMS key ID for processing job outputs
            max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
            sagemaker_session=sagemaker.Session(),  # session object
            tags=None)                              # a list of key-value pairs

    nlp_score_processor.calculate(
        nlp_scorer_config,
        "mdna",                                                                                           # input column
        's3://{}/{}/{}/{}'.format(bucket, sec_processed_folder, 'output', 'dataset_10k_10q.csv'),         # input from s3 bucket
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),                                   # output s3 prefix (both bucket and folder names are required)
        'all_scores.csv'                                                                                  # output file name
    )

The multimodal dataframe after the NLP scoring has completed is shown
below.

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'all_scores.csv'), 'all_scores.csv')
    all_scores = pd.read_csv('all_scores.csv')
    all_scores

Example 2
^^^^^^^^^

The following example shows how to set custom word lists for
``POSITIVE`` and ``NEGATIVE`` score types. The processing job scores
only for the two score types.

.. code:: ipython3

    %%time
    import smjsindustry
    from smjsindustry import NLPScoreType, NLPSCORE_NO_WORD_LIST
    from smjsindustry import NLPScorer
    from smjsindustry import NLPScorerConfig


    custom_positive_word_list = ['good', 'great', 'nice', 'accomplish', 'accept', 'agree', 'believe', 'genius', 'impressive']
    custom_negative_word_list = ['bad', 'broken', 'deny', 'damage', 'disease', 'guilty', 'injure', 'negate', 'pain', 'reject']

    score_type_pos = NLPScoreType(NLPScoreType.POSITIVE, custom_positive_word_list)
    score_type_neg = NLPScoreType(NLPScoreType.NEGATIVE, custom_negative_word_list)

    score_type_list = [score_type_pos, score_type_neg]

    nlp_scorer_config = NLPScorerConfig(score_type_list)

    nlp_score_processor = NLPScorer(
            sagemaker.get_execution_role(),         # loading job execution role
            1,                                      # instances number, limit varies with instance type
            'ml.c5.18xlarge',                       # ec2 instance type to run the loading job
            volume_size_in_gb=30,                   # size in GB of the EBS volume to use
            volume_kms_key=None,                    # KMS key for the processing volume
            output_kms_key=None,                    # KMS key ID for processing job outputs
            max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
            sagemaker_session=sagemaker.Session(),  # session object
            tags=None)                              # a list of key-value pairs

    nlp_score_processor.calculate(
        nlp_scorer_config,
        "mdna",                                                                                           # input column
        's3://{}/{}/{}/{}'.format(bucket, sec_processed_folder, 'output', 'dataset_10k_10q.csv'),         # input from s3 bucket
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),                                   # output s3 prefix (both bucket and folder names are required)
        'scores_custom_word_list.csv'                                                                     # output file name
    )

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'scores_custom_word_list.csv'), 'scores_custom_word_list.csv')
    scores = pd.read_csv('scores_custom_word_list.csv')
    scores

Example 3
^^^^^^^^^

The following example shows how It might take about 30 minutes to run
the following processing job.

.. code:: ipython3

    %%time
    import smjsindustry
    from smjsindustry import NLPScoreType, NLPSCORE_NO_WORD_LIST
    from smjsindustry import NLPScorer
    from smjsindustry import NLPScorerConfig

    societal = pd.read_csv('societal_words.csv', header=None)
    societal_word_list = societal[0].tolist()
    score_type_societal = NLPScoreType('societal', societal_word_list)

    score_type_list = [score_type_societal]

    nlp_scorer_config = NLPScorerConfig(score_type_list)

    nlp_score_processor = NLPScorer(
            sagemaker.get_execution_role(),         # loading job execution role
            1,                                      # instances number, limit varies with instance type
            'ml.c5.18xlarge',                       # ec2 instance type to run the loading job
            volume_size_in_gb=30,                   # size in GB of the EBS volume to use
            volume_kms_key=None,                    # KMS key for the processing volume
            output_kms_key=None,                    # KMS key ID for processing job outputs
            max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
            sagemaker_session=sagemaker.Session(),  # session object
            tags=None)                              # a list of key-value pairs

    nlp_score_processor.calculate(
        nlp_scorer_config,
        "text",                                                                       # input column
        "dataset_10k_10q.csv",                                                        # input file path
        's3://{}/{}/{}'.format(bucket, sec_processed_folder, 'output'),               # output s3 prefix (both bucket and folder names are required)
        'scores_custom_score.csv'                                                     # output file name
    )


.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(sec_processed_folder, 'output', 'scores_custom_score.csv'), 'scores_custom_score.csv')
    custom_scores = pd.read_csv('scores_custom_score.csv')
    custom_scores

Summary
-------

This notebook showed how to:

1. Retrieve parsed plain text of various SEC filings in one API call,
   stored as a CSV file and represented in dataframes.

2. Add columns to the dataframe for different summaries.

3. Score the text column using the ``NLPScorer`` processor for text
   attributes, such as positivity, negativity, and litigiousness, using
   the default word list or custom word lists.

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

Further Supports
----------------

The `SEC filings retrieval API
operations <https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/smjsindustry.finance.data_loader.html>`__
we introduced at the beginning of this example notebook also download
and parse other SEC forms, such as 495, 497, 497K, S-3ASR, and N-1A. If
you need further support for any other types of finance documents, reach
out to the SageMaker JumpStart team through `AWS
Support <https://console.aws.amazon.com/support/>`__ or `AWS Developer
Forums for Amazon
SageMaker <https://forums.aws.amazon.com/forum.jspa?forumID=285>`__.

Reference
---------

1. `What’s New
   post <https://aws.amazon.com/about-aws/whats-new/2021/09/amazon-sagemaker-jumpstart-multimodal-financial-analysis-tools/>`__

2. Blogs:

   -  `Use SEC text for ratings classification using multimodal ML in
      Amazon SageMaker
      JumpStart <https://aws.amazon.com/blogs/machine-learning/use-sec-text-for-ratings-classification-using-multimodal-ml-in-amazon-sagemaker-jumpstart/>`__
   -  `Use pre-trained financial language models for transfer learning
      in Amazon SageMaker
      JumpStart <https://aws.amazon.com/blogs/machine-learning/use-pre-trained-financial-language-models-for-transfer-learning-in-amazon-sagemaker-jumpstart/>`__

3. Documentation and links to the SageMaker JumpStart Industry Python
   SDK:

   -  ReadTheDocs:
      https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/index.html
   -  PyPI: https://pypi.org/project/smjsindustry/
   -  GitHub Repository:
      https://github.com/aws/sagemaker-jumpstart-industry-pack/
   -  Official SageMaker Developer Guide:
      https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart-industry.html

Licence
-------

The SageMaker JumpStart Industry product and its related materials are
under the `Legal License
Terms <https://jumpstart-cache-alpha-us-west-2.s3.us-west-2.amazonaws.com/smfinance-notebook-dependency/legal_file.txt>`__.
