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

Dashboarding SEC Text for Financial NLP
=======================================

The U.S. Securities and Exchange Commission (SEC) filings are widely
used in finance. Companies file the SEC filings to notify the world
about their business conditions and the future outlook of the companies.
Because of the potential predictive values, the SEC filings are good
sources of information for workers in finance, ranging from individual
investors to executives of large financial corporations. These filings
are publicly available to all investors.

In this example notebook, we focus on the following three types of SEC
filings: 10-Ks, 10-Qs, and 8-Ks.

-  `10-Ks <https://www.investopedia.com/terms/1/10-k.asp>`__ - Annual
   reports of companies(and will be quite detailed)
-  `10-Qs <https://www.investopedia.com/terms/1/10q.asp>`__ - Quarterly
   reports, except in the quarter in which a 10K is filed (and are less
   detailed then 10-Ks)
-  `8-Ks <https://www.investopedia.com/terms/1/8-k.asp>`__ - Filed at
   every instance when there is a change in business conditions that is
   material and needs to be reported. This means that there can be
   multiple 8-Ks filed throughout the fiscal year.

The functionality of SageMaker JumpStart Industry will be presented
throughout the notebook, which provides an overall dashboard to
visualize the three types of filings with various analyses. We can
append several standard financial characteristics, such as *Analyst
Recommendation Mean* and *Return on Equity*, but one interesting part of
the dashboard is *attribute scoring*. Using word lists derived from
natural language processing (NLP) techniques, we will score the actual
texts of these filings for a number of characteristics, such as risk,
uncertainty, and positivity, as word proportions, providing simple,
accessible numbers to represent these traits. Using this dashboard,
anybody can pull up information and related statistics about any
companies they have interest in, and digest it in a simple, useful way.

This notebook goes through the following steps to demonstrate how to
extract texts from specific sections in SEC filings, score the texts,
and summarize them.

1. Retrieve and parse 10-K, 10-Q, 8-K filings. Retrieving these filings
   from SEC’s EDGAR service is complicated, and parsing these forms into
   plain text for further analysis can be time consuming. We provide the
   `SageMaker JumpStart Industry Python
   SDK <https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/index.html>`__
   to create a curated dataset in a *single API call*.
2. Create separate dataframes for each of the three types of forms,
   along with separate columns for each extracted section.
3. Combine two or more sections of the 10-K forms and shows how to use
   the NLP scoring API to add numerical values to the columns for the
   text of these columns. The column is called ``text2score``.
4. Add a column with a summary of the ``text2score`` column.
5. Prepare the final dataframe that can be used as input for a
   dashboard.

One of the features of this notebook helps break long SEC filings into
separate sections, each of which deals with different aspects of a
company’s reporting. The goal of this example notebook is to make
accessing and processing texts from SEC filing easy for investors and
training their algorithms.

   **Important**: This example notebook is for demonstrative purposes
   only. It is not financial advice and should not be relied on as
   financial or investment advice.

Financial NLP
-------------

Financial NLP is one of the rapidly increasing use cases of ML in
industry. To find more discussion about this, see the following survey
paper: `Deep Learning for Financial Applications: A
Survey <https://arxiv.org/abs/2002.05786>`__. The starting point for a
vast amount of financial NLP is about extracting and processing texts in
SEC filings. The SEC filings report different types of information
related to various events involving companies. To find a complete list
of SEC forms, see `Forms List <https://www.sec.gov/forms>`__.

The SEC filings are widely used by financial services and companies as a
source of information about companies in order to make trading, lending,
investment, and risk management decisions. They contain forward-looking
information that helps with forecasts and are written with a view to the
future. In addition, in recent times, the value of historical
time-series data has degraded, since economies have been structurally
transformed by trade wars, pandemics, and political upheavals.
Therefore, text as a source of forward-looking information has been
increasing in relevance.

There has been an exponential growth in downloads of SEC filings. See
`How to Talk When a Machine is Listening: Corporate Disclosure in the
Age of AI <https://www.nber.org/papers/w27950>`__; this paper reports
that the number of machine downloads of corporate 10-K and 10-Q filings
increased from 360,861 in 2003 to 165,318,719 in 2016.

A vast body of academic and practitioner research that is based on
financial text, a significant portion of which is based on SEC filings.
A recent review article summarizing this work is `Textual Analysis in
Finance
(2020) <https://www.annualreviews.org/doi/abs/10.1146/annurev-financial-012820-032249>`__.

This notebook describes how a user can quickly retrieve a set of forms,
break them into sections, score texts in each section using pre-defined
word lists, and prepare a dashboard to filter the data.

SageMaker Studio Kernel Setup
-----------------------------

The recommended kernel is **Python 3 (Data Science)**.

*DO NOT* use the **Python 3 (SageMaker JumpStart Data Science 1.0)**
kernel because there are some differences in preinstalled dependencies.

For the instance type, using a larger instance with sufficient memory
can be helpful to download the following materials.

Load SDK and Helper Scripts
---------------------------

The following code cell downloads the ```smjsindustry``
SDK <https://pypi.org/project/smjsindustry/>`__ and helper scripts from
the S3 buckets prepared by SageMaker JumpStart Industry. You will learn
how to use the ``smjsindustry`` SDK which contains various APIs to
curate SEC datasets. The dataset in this example was synthetically
generated using the ``smjsindustry`` package’s SEC Forms Retrieval tool.
For more information, see the `SageMaker JumpStart Industry Python SDK
documentation <https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/notebooks/index.html>`__.

   **Important**: This example notebook uses data obtained from the SEC
   EDGAR database. You are responsible for complying with EDGAR’s access
   terms and conditions located in the `Accessing EDGAR
   Data <https://www.sec.gov/os/accessing-edgar-data>`__ page.

.. code:: ipython3

    # Download scripts from S3
    notebook_artifact_bucket = 'jumpstart-cache-alpha-us-west-2'
    notebook_sdk_prefix = 'smfinance-notebook-dependency/smjsindustry'
    notebook_script_prefix = 'smfinance-notebook-data/sec-dashboard'
    
    # Download smjsindustry SDK
    sdk_bucket = f's3://{notebook_artifact_bucket}/{notebook_sdk_prefix}'
    !aws s3 sync $sdk_bucket ./
    
    # Download helper scripts
    scripts_bucket = f's3://{notebook_artifact_bucket}/{notebook_script_prefix}'
    !aws s3 sync $scripts_bucket ./sec-dashboard

Install the ``smjsindustry`` library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We deliver APIs through the ``smjsindustry`` client library. The first
step requires pip installing a Python package that interacts with a
SageMaker processing container. The retrieval, parsing, transforming,
and scoring of text is a complex process and uses many different
algorithms and packages. To make this seamless and stable for the user,
the functionality is packaged into an collection of APIs. For
installation and maintenance of the workflow, this approach reduces your
eﬀort to a pip install followed by a single API call.

.. code:: ipython3

    # Install smjsindustry SDK
    !pip install --no-index smjsindustry-1.0.0-py3-none-any.whl

Load the functions for extracting the “Item” sections from the forms
--------------------------------------------------------------------

We created various helper functions to enable sectioning the SEC forms.
These functions do take some time to load.

.. code:: ipython3

    %run sec-dashboard/SEC_Section_Extraction_Functions.ipynb

The next block loads packages for using the AWS resources, SageMaker
features, and SageMaker JumpStart Industry SDK.

.. code:: ipython3

    %pylab inline
    import boto3
    import pandas as pd
    import sagemaker
    pd.get_option("display.max_columns", None)
    
    import smjsindustry
    from smjsindustry.finance import utils
    from smjsindustry import NLPScoreType, NLPSCORE_NO_WORD_LIST
    from smjsindustry import NLPScorerConfig, JaccardSummarizerConfig, KMedoidsSummarizerConfig
    from smjsindustry import Summarizer, NLPScorer
    from smjsindustry.finance.processor import DataLoader, SECXMLFilingParser
    from smjsindustry.finance.processor_config import EDGARDataSetConfig

Next, we import required packages and load the S3 bucket from the
SageMaker session, as shown below.

.. code:: ipython3

    # Prepare the SageMaker session's default S3 bucket and a folder to store processed data
    session = sagemaker.Session()
    bucket = session.default_bucket()
    secdashboard_processed_folder='jumpstart_industry_secdashboard_processed'

Download the filings you wish to work with
------------------------------------------

Downloading SEC filings is done from the SEC’s Electronic Data
Gathering, Analysis, and Retrieval (EDGAR) website, which provides open
data access. EDGAR is the primary system under the U.S. Securities And
Exchange Commission (SEC) for companies and others submitting documents
under the Securities Act of 1933, the Securities Exchange Act of 1934,
the Trust Indenture Act of 1939, and the Investment Company Act of 1940.
EDGAR contains millions of company and individual filings. The system
processes about 3,000 filings per day, serves up 3,000 terabytes of data
to the public annually, and accommodates 40,000 new filers per year on
average. Below we provide a simple *one*-API call that will create a
dataset of plain text filings in a few lines of code, for any period of
time and for a large number of tickers.

We have wrapped the extraction functionality into a SageMaker processing
container and provide this notebook to enable users to download a
dataset of filings with meta data such as dates and parsed plain text
that can then be used for machine learning using other SageMaker tools.
Users only need to specify a date range and a list of ticker symbols and
this API will do the rest.

The extracted dataframe is written to S3 storage and to the local
notebook instance.

The API below specifies the machine to be used and the volume size. It
also specifies the tickers or CIK codes for the companies to be covered,
as well as the 3 form types (10-K, 10-Q, 8-K) to be retrieved. The data
range is also specified as well as the filename (CSV) where the
retrieved filings will be stored.

The API is in 3 parts:

1. Set up a dataset configuration (an ``EDGARDataSetConfig`` object).
   This specifies (i) the tickers or SEC CIK codes for the companies
   whose forms are being extracted; (ii) the SEC forms types (in this
   case 10-K, 10-Q, 8-K); (iii) date range of forms by filing date, (iv)
   the output CSV file and S3 bucket to store the dataset.
2. Set up a data loader object (a ``DataLoader`` object). The middle
   section shows how to assign system resources and has default values
   in place.
3. Run the data loader (``data_loader.load``).

This initiates a processing job running in a SageMaker container.

   **Important**: This example notebook uses data obtained from the SEC
   EDGAR database. You are responsible for complying with EDGAR’s access
   terms and conditions located in the `Accessing EDGAR
   Data <https://www.sec.gov/os/accessing-edgar-data>`__ page.

.. code:: ipython3

    %%time
    
    dataset_config = EDGARDataSetConfig(
        tickers_or_ciks=['amzn', 'goog', '27904', 'fb', 'msft', 'uber', 'nflx'],  # list of stock tickers or CIKs
        form_types=['10-K', '10-Q', '8-K'],              # list of SEC form types
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
        's3://{}/{}/{}'.format(bucket, secdashboard_processed_folder, 'output'),                  # output s3 prefix (both bucket and folder names are required)
        'dataset_10k_10q_8k_2019_2021.csv',                                                       # output file name
        wait=True,
        logs=True)

Copy the file into Studio from the s3 bucket
--------------------------------------------

We can examine the dataframe that was constructed by the API.

.. code:: ipython3

    client = boto3.client('s3')
    client.download_file(bucket, '{}/{}/{}'.format(secdashboard_processed_folder, 'output', 'dataset_10k_10q_8k_2019_2021.csv'), 'dataset_10k_10q_8k_2019_2021.csv')

See how a complete dataset was prepared. Altogether, a few hundred forms
were retrieved across tickers and the three types of SEC form.

.. code:: ipython3

    df_forms = pd.read_csv('dataset_10k_10q_8k_2019_2021.csv')
    df_forms

Here is a breakdown of the few hundred forms by **ticker** and
**form_type**.

.. code:: ipython3

    df_forms.groupby(['ticker','form_type']).count().reset_index()

Create the dataframe for the extracted item sections from the 10-K filings
--------------------------------------------------------------------------

In this section, we break the various sections of the 10-K filings into
separate columns of the extracted dataframe.

1. Take a subset of the dataframe by specifying
   ``df.form_type == "10-K"``.
2. Extract the sections for each 10-K filing and put them in columns in
   a separate dataframe.
3. Merge this dataframe with the dataframe from Step 1.

You can examine the cells in the dataframe below to see the text from
each section.

.. code:: ipython3

    df = pd.read_csv('dataset_10k_10q_8k_2019_2021.csv')
    df_10K = df[df.form_type == "10-K"]

.. code:: ipython3

    # Construct the DataFrame row by row.
    items_10K = pd.DataFrame(columns = columns_10K, dtype=object)
    # for i in range(len(df)):
    for i in df_10K.index:
        form_text = df_10K.text[i]
        item_iter = get_form_items(form_text, "10-K")
        items_10K.loc[i] = items_to_df_row(item_iter, columns_10K, "10-K")

.. code:: ipython3

    items_10K.rename(columns=header_mappings_10K, inplace=True)
    # items_10K.head(10)

.. code:: ipython3

    df_10K = pd.merge(df_10K, items_10K, left_index=True, right_index=True)
    df_10K.head(10)

Let’s take a look at the text in one of the columns to see that there is
clean, parsed, plain text provided by the API:

.. code:: ipython3

    print(df_10K["Risk Factors"][138])

Similarly, we can create the dataframe for the extracted item sections from the 10-Q filings
--------------------------------------------------------------------------------------------

1. Take a subset of the dataframe by specifying
   ``df.form_type == "10-Q"``.
2. Extract the sections for each 10-Q filing and put them in columns in
   a separate dataframe.
3. Merge this dataframe with the dataframe from 1.

.. code:: ipython3

    df = pd.read_csv('dataset_10k_10q_8k_2019_2021.csv')
    df_10Q = df[df.form_type == "10-Q"]

.. code:: ipython3

    # Construct the DataFrame row by row.
    items_10Q = pd.DataFrame(columns=columns_10Q, dtype=object)
    # for i in range(len(df)):
    for i in df_10Q.index:
        form_text = df_10Q.text[i]
        item_iter = get_form_items(form_text, "10-Q")
        items_10Q.loc[i] = items_to_df_row(item_iter, columns_10Q, "10-Q")

.. code:: ipython3

    items_10Q.rename(columns=header_mappings_10Q, inplace=True)
    # items_10Q.head(10)

.. code:: ipython3

    df_10Q = pd.merge(df_10Q, items_10Q, left_index=True, right_index=True)
    df_10Q.head(10)

Create the dataframe for the extracted item sections from the 8-K filings
-------------------------------------------------------------------------

1. Take a subset of the dataframe by specifying
   ``df.form_type == "8-K"``.
2. Extract the sections for each 8-K filing and put them in columns in a
   separate dataframe.
3. Merge this dataframe with the dataframe from Step 1.

.. code:: ipython3

    df = pd.read_csv('dataset_10k_10q_8k_2019_2021.csv')
    df_8K = df[df.form_type == "8-K"]

.. code:: ipython3

    # Construct the DataFrame row by row.
    items_8K = pd.DataFrame(columns=columns_8K, dtype=object)
    # for i in range(len(df)):
    for i in df_8K.index:
        form_text = df_8K.text[i]
        item_iter = get_form_items(form_text, "8-K")
        items_8K.loc[i] = items_to_df_row(item_iter, columns_8K, "8-K")

.. code:: ipython3

    items_8K.rename(columns=header_mappings_8K, inplace=True)
    # items_8K.head(10)

.. code:: ipython3

    df_8K = pd.merge(df_8K, items_8K, left_index=True, right_index=True)
    # df_8K

.. code:: ipython3

    df1 = df_8K.copy()
    df1 = df1.mask(df1.apply(lambda x: x.str.len().lt(1)))
    df1

Summary table of section counts
-------------------------------

.. code:: ipython3

    df1 = df1.groupby('ticker').count()
    df1[df1.columns[5:]]

NLP scoring of the 10-K forms for specific sections
---------------------------------------------------

Financial text has been scored using word lists for some time. See the
paper `“Textual Analysis in
Finance” <https://www.investopedia.com/terms/1/8-k.asp>`__ for a
comprehensive review.

The ``smjsindustry`` library provides 11 NLP score types by default:
``positive``, ``negative``, ``litigious``, ``polarity``, ``risk``,
``readability``, ``fraud``, ``safe``, ``certainty``, ``uncertainty``,
and ``sentiment``. Each score (except readability and sentiment) has its
word list, which is used for scanning and matching with an input text
dataset.

NLP scoring delivers a score as the fraction of words in a document that
are in the relevant scoring word lists. Users can provide their own
custom word list to calculate the NLP scores. Some scores like
readability use standard formulae such as the Gunning-Fog score.
Sentiment scores are based on the
`VADER <https://pypi.org/project/vaderSentiment/>`__ library.

These NLP scores are added as new numerical columns to the text
dataframe; this creates a multimodal dataframe, which is a mixture of
tabular data and longform text, called **TabText**. When submitting this
multimodal dataframe for ML, it is a good idea to normalize the columns
of NLP scores (usually with standard normalization or min-max scaling).

Any chosen text column can be scored automatically using the tools in
SageMaker JumpStart. We demonstrate this below.

As an example, we combine the MD&A section (Item 7) and the Risk section
(Item 7A), and then apply NLP scoring. We compute 11 additional columns
for various types of scores.

Since the size of the SEC filings text can be very large, NLP scoring is
computationally time-consuming, so we have built the API to enable
distribution of this task across multiple machines. In the API, users
can choose the number and type of machine instances they want to run NLP
scoring on in distributed fashion.

To begin, earmark the text for NLP scoring by creating a new column that
combines two columns into a single column called ``text2score``. A new
file is saved in the Amazon S3 bucket.

.. code:: ipython3

    df_10K["text2score"] = [i+' '+j for i,j in zip(df_10K["Management’s Discussion and Analysis of Financial Condition and Results of Operations"],
                            df_10K["Quantitative and Qualitative Disclosures about Market Risk"])]
    df_10K[['ticker','text2score']].to_csv('text2score.csv', index=False)

.. code:: ipython3

    client.upload_file('text2score.csv', bucket, '{}/{}/{}'.format(secdashboard_processed_folder, 'output', 'text2score.csv'))

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
   which contain anywhere from 20,000-100,000 words. Matching to word
   lists (usually 200 words or more) can be time-consuming.
7. VPC mode is supported in this API.

**Input**

The input to the API requires (i) what NLP scores to be generated, each
one resulting in a new column in the dataframe; (ii) specification of
system resources, i.e., number and type of machine instances to be used;
(iii) the s3 bucket and filename in which to store the enhanced
dataframe as a CSV file; (iv) a section that kicks off the API.

**Output**

The output filename used in the example below is ``all_scores.csv``, but
yiou can change this to any other filename. It’s stored in the S3 bucket
and then, as shown in the following code, we copy it into Studio here to
process it into a dashboard.

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
            'ml.c5.9xlarge',                       # ec2 instance type to run the loading job
            volume_size_in_gb=30,                   # size in GB of the EBS volume to use
            volume_kms_key=None,                    # KMS key for the processing volume
            output_kms_key=None,                    # KMS key ID for processing job outputs
            max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
            sagemaker_session=sagemaker.Session(),  # session object
            tags=None)                              # a list of key-value pairs
    
    nlp_score_processor.calculate(
        nlp_scorer_config, 
        "text2score",                                                                                              # input column
        's3://{}/{}/{}/{}'.format(bucket, secdashboard_processed_folder, 'output', 'text2score.csv'),              # input from s3 bucket
        's3://{}/{}/{}'.format(bucket, secdashboard_processed_folder, 'output'),                                   # output s3 prefix (both bucket and folder names are required)
        'all_scores.csv'                                                                                           # output file name
    )

.. code:: ipython3

    client.download_file(bucket, '{}/{}/{}'.format(secdashboard_processed_folder, 'output', 'all_scores.csv'), 'all_scores.csv')

Stock Screener based on NLP scores
----------------------------------

Once we have added columns for all the NLP scores, we can then screen
the table for companies with high scores on any of the attributes. See
the table below.

.. code:: ipython3

    qdf = pd.read_csv('all_scores.csv')
    qdf.head()

Add a column with summaries of the text being scored
----------------------------------------------------

We can further enhance the dataframe with summaries of the target text
column. As an example, we used the abstractive summarizer from Hugging
Face. Since this summarizer can only accomodate roughly 300 words of
text, it’s not directly applicable to our text, which is much longer
(thousands of words). Therefore, we applied the Hugging Face summarizer
to groups of paragraphs and pulled it all together to make a single
summary. We created a helper function ``fullSummary`` that is called in
the code below to create a summary of each document in the column
``text2score``.

Notice that the output dataframe is now extended with an additional
summary column.

*Note*: An abstractive summarizer restructures the text and loses the
original sentences. This is in contrast to an extractive summarizer,
which retain the original sentence structure.

Summarization is time consuming and this code block takes time. We do
the first 5 documents in the ``text2score`` column to illustrate.

.. code:: ipython3

    %%time
    qdf['summary'] = ''
    for i in range(5):
        qdf.loc[i,'summary'] = fullSummary(qdf.loc[i,'text2score'])
        print(i, end='..')

Examine one of the summaries.

.. code:: ipython3

    i = 2
    print(qdf.summary[i])
    print('---------------')
    print(qdf.text2score[i])

Store the curated dataset
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    qsf = qdf.drop(['text2score'], axis=1)
    qsf.to_csv('stock_sec_scores.csv', index=False)

To complete this example notebook, we provide two artifacts that may be
included in a dashboard: 1. Creating an interactive datatable so that a
non-technical user may sort and filter the rows of the curated
dataframe. 2. Visualizing the differences in documents by NLP scores
using radar plots.

This is shown next.

Create an interactive dashboard
-------------------------------

Using the generated CSV file, you can construct an interactive screening
dashboard.

Run from an R script to construct the dashboard. All you need is just
this single block of code below. it will create a browser enabled
interactive data table, and save it in a file title
``SEC_dashboard.html``. You may open it in a browser.

.. code:: ipython3

    import subprocess
    
    ret_code=subprocess.call(['/usr/bin/Rscript', 'sec-dashboard/Dashboard.R'])

After the notebook finishes running, open the ``SEC_Dashboard.html``
file that was created. You might need to click ``Trust HTML`` at the
upper left corner to see the filterable table and the content of it. The
following screenshot shows an example of the filterable table.

.. code:: ipython3

    from IPython.display import Image
    Image("sec-dashboard/dashboard.png", width=800, height=600)

Visualizing the text through the NLP scores
-------------------------------------------

The following vizualition function shows how to create a *radar plot* to
compare two SEC filings using their normalized NLP scores. The scores
are normalized using min-max scaling on each NLP score. The radar plot
is useful because it shows the overlap (and consequently, the
difference) between the documents.

.. code:: ipython3

    ## Read in the scores
    scores = pd.read_csv('stock_sec_scores.csv')
    
    # Choose whichever filings you want to compare for the 2nd and 3rd parameter
    createRadarChart(scores, 2, 11)

Further support
---------------

The `SEC filings retrieval API
operations <https://sagemaker-jumpstart-industry-pack.readthedocs.io/en/latest/smjsindustry.finance.data_loader.html>`__
we introduced at the beginning of this example notebook also download
and parse other SEC forms, such as 495, 497, 497K, S-3ASR, and N-1A. If
you need further support for any other types of finance documents, reach
out to the SageMaker JumpStart team through `AWS
Support <https://console.aws.amazon.com/support/>`__ or `AWS Developer
Forums for Amazon
SageMaker <https://forums.aws.amazon.com/forum.jspa?forumID=285>`__.

References
----------

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

Licences
--------

The SageMaker JumpStart Industry product and its related materials are
under the `Legal License
Terms <https://jumpstart-cache-alpha-us-west-2.s3-us-west-2.amazonaws.com/smfinance-notebook-dependency/licenses.txt>`__.

   **Important**: (1) This notebook is for demonstrative purposes only.
   It is not financial advice and should not be relied on as financial
   or investment advice. (2) This notebook uses data obtained from the
   SEC EDGAR database. You are responsible for complying with EDGAR’s
   `access terms and
   conditions <https://www.sec.gov/os/accessing-edgar-data>`__.

This notebook utilizes certain third-party open source software packages
at install-time or run-time (“External Dependencies”) that are subject
to copyleft license terms you must accept in order to use it. If you do
not accept all of the applicable license terms, you should not use the
notebook. We recommend that you consult your company’s open source
approval policy before proceeding. Provided below is a list of External
Dependencies and the applicable license identification as indicated by
the documentation associated with the External Dependencies as of
Amazon’s most recent review. - R v3.5.2: GPLv3 license
(https://www.gnu.org/licenses/gpl-3.0.html) - DT v0.19.1: GPLv3 license
(https://github.com/rstudio/DT/blob/master/LICENSE)

THIS INFORMATION IS PROVIDED FOR CONVENIENCE ONLY. AMAZON DOES NOT
PROMISE THAT THE LIST OR THE APPLICABLE TERMS AND CONDITIONS ARE
COMPLETE, ACCURATE, OR UP-TO-DATE, AND AMAZON WILL HAVE NO LIABILITY FOR
ANY INACCURACIES. YOU SHOULD CONSULT THE DOWNLOAD SITES FOR THE EXTERNAL
DEPENDENCIES FOR THE MOST COMPLETE AND UP-TO-DATE LICENSING INFORMATION.

YOUR USE OF THE EXTERNAL DEPENDENCIES IS AT YOUR SOLE RISK. IN NO EVENT
WILL AMAZON BE LIABLE FOR ANY DAMAGES, INCLUDING WITHOUT LIMITATION ANY
DIRECT, INDIRECT, CONSEQUENTIAL, SPECIAL, INCIDENTAL, OR PUNITIVE
DAMAGES (INCLUDING FOR ANY LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST
PROFITS OR DATA, OR COMPUTER FAILURE OR MALFUNCTION) ARISING FROM OR
RELATING TO THE EXTERNAL DEPENDENCIES, HOWEVER CAUSED AND REGARDLESS OF
THE THEORY OF LIABILITY, EVEN IF AMAZON HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES. THESE LIMITATIONS AND DISCLAIMERS APPLY
EXCEPT TO THE EXTENT PROHIBITED BY APPLICABLE LAW.

