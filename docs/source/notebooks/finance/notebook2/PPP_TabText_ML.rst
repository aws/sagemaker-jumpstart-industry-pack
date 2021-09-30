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

Machine Learning on a TabText Dataframe
=======================================

An Example Based on the Paycheck Protection Program
---------------------------------------------------

The Paycheck Protection Program (PPP) was created by the U.S. government
to enable employers struggling with COVID-related business adversities
to make payments to their employees. For more information, see the
`Paycheck Protection
Program <https://www.sba.gov/funding-programs/loans/coronavirus-relief-options/paycheck-protection-program>`__.
In this example notebook, you’ll learn how to run a machine learning
model on a sample of companies in the program over the first two
quarters of 2020.

In this notebook, we take U.S Securities and Exchange Commission (SEC)
filing data from some of the companies that partook of the loans under
this program. We demonstrate how to merge the SEC filing data (text
data) with stock price data (tabular data) using the SageMaker JumpStart
Industry Python SDK. The ``build_tabText`` class of the library helps
merge text dataframes with numeric dataframes to create a multimodal
dataframe for machine learning.

| A subset of the list of tickers of firms that took PPP loans obtained
  from authors of the following paper:
| - Balyuk, T., Prabhala, N. and Puri, M. (November 2020, revised June
  2021), `Indirect Costs of Government Aid and Intermediary Supply
  Effects: Lessons from the Paycheck Protection
  Program <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3735682>`__.
  NBER Working Paper No. w28114, Available at SSRN:
  https://ssrn.com/abstract=3735682.

The PPP program was created to help companies experiencing financial
hardship and may have otherwise been unable to make payroll. To the
degree companies not experiencing financial constraints took PPP money,
they may have experienced a decrease in share value and returned the
money to recoup value.

We are interested in seeing if an ML model is able to detect whether the
text of filings of companies that returned PPP money is different from
that of companies that retained PPP money.


General Steps
~~~~~~~~~~~~~

This notebook takes the following steps:

1. Read in over 400 tickers of companies that took the PPP loans.
2. Read in the 10-K, 10-Q, and 8K filings for all paycheck protection
   tickers during Q1 and Q2 of 2020. Texts in the SEC filings are loaded
   using the ``smjsindustry.DataLoader`` class.
3. Load a synthetic time series data of daily stock prices for the given
   tickers during Q1 and Q2 of 2020. Convert prices to returns. The
   simulated data is generated to be correlated with appropriate labels
   so that it can be meaningful. An analogous exercise with true data
   yields similar results.
4. Merge text and tabular datasets using the
   ``smjsindustry.build_tabText`` API.
5. Conduct machine learning analysis to obtain a baseline accuracy.
6. Build an `AutoGluon <https://auto.gluon.ai/stable/index.html>`__
   model to analyze how stock prices and texts in the SEC filings are
   related to each company’s decision to accept or return the money.
   This notebook shows how to flag all filings of companies that return
   the money with a 1 and the filings of companies that do not return
   the money with a 0. A good fit to the data implies the model can
   distinguish companies into two categories: the ones that return PPP
   funding versus those that do not based on the text.

Objective
---------

The goal in this notebook is to fit an ML model to the data on companies
that partook of funding from the Paycheck Protection Program (PPP) and
to study how stock prices and returns and text from the SEC forms are
related to their decisions to return the money.

The PPP program is reported in each company’s 8-K, an SEC filing which
is required when a public company experiences a material change in
business conditions. In addition to a 8-K filing, 10-K and 10-Q filings,
which present a comprehensive summary of a company’s financial
performance, are also used as source of inputs for this study. The stock
data is *synthetically generated* to be correlated with the labels. You
can repeat this exercise with actual data as needed.

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
    notebook_data_prefix = 'smfinance-notebook-data/ppp'
    notebook_sdk_prefix = 'smfinance-notebook-dependency/smjsindustry'
    notebook_autogluon_prefix = 'smfinance-notebook-dependency/autogluon'

.. code:: ipython3

    data_bucket = f's3://{notebook_artifact_bucket}/{notebook_data_prefix}'
    !aws s3 sync $data_bucket ./

Install the SageMaker JumpStart Industry Python SDK and dependencies
that are needed for machine learning, because they are not available as
defaults in Studio.

.. code:: ipython3

    sdk_bucket = f's3://{notebook_artifact_bucket}/{notebook_sdk_prefix}'
    !aws s3 sync $sdk_bucket ./

    !pip install --no-index smjsindustry-1.0.0-py3-none-any.whl

Step 1: Read in the Tickers
---------------------------

Over 400 tickers are used for this study.

.. code:: ipython3

    %pylab inline
    import pandas as pd
    import os

    ppp_tickers = pd.read_excel("ppp_tickers.xlsx", index_col = None, sheet_name=0)
    print("Number of PPP tickers =", ppp_tickers.shape[0])
    ticker_list = list(set(ppp_tickers.ticker))
    ppp_tickers.head()

Step 2: Read in the SEC Forms Filed by These Companies
------------------------------------------------------

1. This notebook retrieves all 10-K/Q, 8-K forms from the SEC servers
   for Q1 and Q2 of 2020. This was done using the SageMaker JumpStart
   Industry Python SDK’s ``DataLoader`` class. For reference, the time
   taken by the data lodaer process was around 30 minutes for curating a
   dataframe of over 4000 filings.
2. There is one 10K/Q form per quarter. These are quarterly reports.
3. There can be multiple 8K forms per quarter, because these are filed
   for material changes in business conditions. Depending on how many
   such events there are, several 8Ks might need to be filed. As you
   will see, this notebook retrieves more than one form per quarter.
4. The dataset was stored in a CSV file named ``ppp_10kq_8k_data.csv``
   (351 MB).

..

   **Important**: This example notebook uses data obtained from the SEC
   EDGAR database. You are responsible for complying with EDGAR’s access
   terms and conditions located in the `Accessing EDGAR
   Data <https://www.sec.gov/os/accessing-edgar-data>`__ page.

.. code:: ipython3

    %%time
    df_sec = pd.read_csv("ppp_10kq_8k_data.csv")  # Text data

.. code:: ipython3

    print("Number of SEC filings: ", df_sec.shape[0])

Step 3: Collect Stock Prices and Convert to Returns
---------------------------------------------------

-  Given the list of tickers, we synthetically generated stock prices
   using simulation of geometric Brownian motion. The stock prices are
   generated to be consistent with the real market data. You can buy
   data for commercial use if needed.
-  Convert the stock prices to returns.

Some tickers might have been delisted since the time of the PPP program.

Read in the PPP stock prices synthetic dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    df_prices = pd.read_csv("ppp_stock_prices_synthetic.csv")
    print('Total number of days for the stock time series: ', df_prices.shape[0])
    print('Total number of stocks: ', df_prices.shape[1])
    df_prices.head()

The following code cell converts the prices into percentage returns.

-  It converts prices into returns.
-  It calls helper function to convert prices to returns.
-  It removes the stock that only has ``NaN`` values, if any.
-  It converts prices to returns using the ``pct_change`` function.

.. code:: ipython3

    def convert_price_to_return(df_prices):
        ticker_list = list(df_prices.columns[1:])
        df_returns = df_prices[ticker_list].pct_change()                  # not using fill_method='ffill'
        df_returns = pd.concat([df_prices.Date, df_returns], axis=1)[1:]  # drop first row as it is NaN
        df_returns = df_returns.reset_index(drop=True)
        return df_returns

    df_returns = convert_price_to_return(df_prices)
    df_returns.dropna(axis=1, how='all', inplace = True)                  # drop columns with partial data
    df_returns.set_index('Date', inplace = True)
    print('Total number of stocks: ', len(list(df_returns.columns[1:])))
    df_returns.head()

Convert the dataframe to CSV and save
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    df_returns.to_csv('ppp_returns.csv', index = True)

Step 4: Merge Text and Tabular Datasets
---------------------------------------

The stock returns and the SEC forms are saved in earlier code blocks
into CSV files. In this step, you’ll learn how to read in the files and
merge the text data with the tabular data.

-  Line up the returns from day -5 before the filing date to day +5
   after the filng date. Including the return on the filing date itself,
   we get 11 days of returns around the filing date.
   Three types of returns are considered here: > **Ret** - stock return
   > **MktRet** - S&P 500 return
   > **NetRet** - difference between ``Ret`` and ``MktRet``
-  Merge the SEC text data and the tabular data with the
   ``build_tabText`` API. We need to see how returns evolve around the
   filing date.

Read in the return data and the text data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time

    df_returns = pd.read_csv("ppp_returns.csv")    # Tabular/numeric data
    df_sec = pd.read_csv("ppp_10kq_8k_data.csv")   # Text data

Define helper functions to create 3 types of returns for 5 days before
and 5 days after the filing date. The functions fill in returns for the
ticker and corresponding S&P return.

.. code:: ipython3

    %%time

    def fillReturn(df_returns, ticker, dt, displacement):
        if np.where(df_returns.columns == ticker)[0].size > 0:
            bwd = list(df_returns[ticker].loc[:dt][-(displacement+1):]) # 5 days before filing plus filing date
            fwd = list(df_returns[ticker].loc[dt:][1:(displacement+1)]) # 5 days after filing
            if len(bwd) < displacement+1:
                bwd = [np.nan]*(displacement+1-len(bwd)) + bwd          # Add NaN at the beginning if less bwd
            if len(fwd) < displacement:
                fwd = fwd + [np.nan]*(displacement-len(fwd))            # Append NaN in the end if less fwd
            return bwd+fwd
        else:
            return [np.nan for idx in range(2*displacement+1)]

    def create_df_5_days_return(df_returns):
        displace = 5
        cols = ['Date', 'ticker', 'Ret-5', 'Ret-4', 'Ret-3', 'Ret-2', 'Ret-1', 'Ret0', 'Ret1', 'Ret2', 'Ret3', 'Ret4', 'Ret5',
                'MktRet-5', 'MktRet-4', 'MktRet-3', 'MktRet-2', 'MktRet-1', 'MktRet0', 'MktRet1', 'MktRet2', 'MktRet3', 'MktRet4', 'MktRet5',
                'NetRet-5', 'NetRet-4', 'NetRet-3', 'NetRet-2', 'NetRet-1', 'NetRet0', 'NetRet1', 'NetRet2','NetRet3', 'NetRet4', 'NetRet5']
        df_trans_dict = {}
        idx = 0
        for ticker in df_returns.columns[1:]:
            for row in range(len(df_returns)):
                dt = df_returns.Date[row]
                rets = fillReturn(df_returns, ticker, dt, displace)
                mkt_rets = fillReturn(df_returns, '^GSPC', dt, displace)
                net_rets = [ a-b for a, b in zip(rets, mkt_rets)]
                row_data = [dt, ticker] + rets + mkt_rets + net_rets
                df_trans_dict[idx] = row_data
                idx += 1
        df_returns_trans = pd.DataFrame.from_dict(df_trans_dict, orient='index', columns = cols)
        return df_returns_trans

    df_returns_trans = create_df_5_days_return(df_returns)
    pd.set_option('display.max_columns', 50)
    df_returns_trans.head(5)

Create a TabText dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code cell calls the ``smjsindustry.build_tabText`` class
to create a multimodal TabText dataframe, merging the tabular data and
the text data together; the dataframe should have the ``Date`` column
and a common column (‘ticker’ in this case) to generate a time series
TabText dataset.

.. code:: ipython3

    %%time
    # Use build_tabText API to merge text and tabular datasets
    from smjsindustry import build_tabText

    tab_text = build_tabText(
            df_sec,
            "ticker",
            "filing_date",
            df_returns_trans,
            "ticker",
            "Date",
            freq='D'
        )

.. code:: ipython3

    tab_text.head()

Write the merged dataframe into a CSV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    tab_text.to_csv("ppp_10kq_8k_stock_data.csv", index=False)

Step 5: Machine Learning Analysis
---------------------------------

Some of these companies subsequently returned the money. Returning the
money results in signaling an improvement in business conditions with a
subsequent uptick in stock prices. Thus, an exercise to predict which
firms would return the money based on their SEC filings is of interest.

| The following code cells prepare the dataset for ML studies with the
  following steps:
| \* It flags all filings of the companies that returned the PPP money
  with a 1 and the others with a 0. Therefore, an ML model fit to these
  labels teases out whether the text for companies that retain PPP money
  is distinguishable from text of companies that return PPP money.

The resultant dataframe from the previous steps is stored as a CSV file
titled ``ppp_model_TabText.csv`` (354 MB). This file contains both text
and numerical columns of data.

Read in the TabText dataframe and get the returned ticker list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    tab_text = pd.read_csv('ppp_10kq_8k_stock_data.csv')

    ppp_tickers_returned = pd.read_excel('ppp_tickers_returned.xlsx', index_col = None, sheet_name=0)
    print("Number of PPP Returned tickers =", ppp_tickers_returned.shape[0])
    ticker_list_returned = list(set(ppp_tickers_returned.ticker))

.. code:: ipython3

    tab_text['returned'] = [1 if j in ticker_list_returned else 0 for j in tab_text['ticker']]

.. code:: ipython3

    tab_text

Add the ``"returned"`` label (1,0) to each row as required
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    tab_text['returned'] = [1 if j in ticker_list_returned else 0 for j in tab_text['ticker']]
    tab_text = tab_text.drop(['Date'], axis=1)
    tab_text.to_csv('ppp_model_TabText.csv', index=False)

You can start examining the mean return in the 5 days before the filing
(-5,0) and 5 days after the filing (0,+5) to see how the firms that
returned the money fared, compared to those that did not return the
money. You’ll learn how the mean excess return (over the S&P return)
between the two groups are calculated.

Read in the TabText dataframe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    df = pd.read_csv("ppp_model_TabText.csv")
    print(df.shape)
    print(df.columns)

Next, the following cell curates the TabText dataframe by creating a
cumulative (net of market) return for the 5 days before the filing
(``df["First5"]``) and the 5 days after the filing (``df["Second5"]``).
You can also see the various feature columns shown in the dataframe as
shown in the following cell.

.. code:: ipython3

    # Add up the returns for days (-5,0) denoted "First5" and days (0,5) denoted second 5
    # Note that it is actually 6 days of returns.
    df["First5"] = df["NetRet-5"] + df["NetRet-4"] + df["NetRet-3"] + df["NetRet-2"] + df["NetRet-1"] + df["NetRet0"]
    df["Second5"] = df["NetRet5"] + df["NetRet4"] + df["NetRet3"] + df["NetRet2"] + df["NetRet1"] + df["NetRet0"]
    df.head()

.. code:: ipython3

    res = df.groupby(['returned']).count()['ticker']
    print(res)
    print("Baseline accuracy =", res[0]/sum(res))

.. code:: ipython3

    df.groupby(['returned']).mean()[["First5","Second5"]]

From the output of the preceding cell, the mean return for the
``"First5"`` set is slightly worse for the ``"returned=0"`` case and the
mean return for the ``"Second5"`` set is higher for the ``"returned=1"``
case. Maybe firms that returned the money were signalling to the market
that they were in good shape and the market rewarded them with a stock
price bounce.

Step 6: Machine Learning on the TabText Dataframe
-------------------------------------------------

In this notebook, an AutoGluon model is used with the SageMaker MXNet
training framework to analyze how leading stock returns for 5 days
(numerical data) and 10-K/Q, 8-K filings (text) are related to each
company’s decision to accept or return the money.

Train an AutoGluon Model for Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, you’ll see how easy it is to undertake a seamless ML on multimodal
data (TabText). In this section, you’ll learn how to use one of the open
source AWS libraries known as AutoGluon, which is a part of the Gluon
NLP family of tools. To learn more, see `GluonNLP: NLP made
easy <https://nlp.gluon.ai/>`__.

In particular, we use the AutoGluon-Tabular model, which is designed for
TabText and has superior performance. For more information about the
model, see `AutoGluon-Tabular: Robust and Accurate AutoML for Structured
Data <https://arxiv.org/abs/2003.06505>`__.

For a quick start, see `Predicting Columns in a Table - Quick
Start <https://auto.gluon.ai/tutorials/tabular_prediction/tabular-quickstart.html>`__.
To find the AutoGluon-Tabular model in AWS Marketplace, see
`AutoGluon-Tabular <https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AutoGluon-Tabular/prodview-n4zf5pmjt7ism>`__.

The AutoGluon-Tabular model processes the data and trains a diverse
ensemble of ML models to create a “predictor” which is able to predict
the ``"returned"`` label in this data. This example uses both return and
text data to build a model.

Create a sample dataset
^^^^^^^^^^^^^^^^^^^^^^^

For demonstration purposes, take a sample from the original dataset to
reduce the time for training.

.. code:: ipython3

    sample_df = pd.concat([df[df["returned"]==1].sample(n=500), df[df["returned"]==0].sample(n=500)]).sample(frac=1)

Save the dataframe into a CSV file.

.. code:: ipython3

    sample_df.to_csv('ppp_model_sample_input.csv', index=False)

Prepare the SageMaker Training Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following cells download installation packages, and create ``lib``
folder and ``requirements.txt`` file to store AutoGluon related
dependencies. These dependencies will be installed in the training
containers. For more information, see `Use third-party
libraries <https://sagemaker.readthedocs.io/en/stable/frameworks/mxnet/using_mxnet.html#use-third-party-libraries>`__
in the *Amazon SageMaker Python SDK documentation*.

Download AutoGluon installation packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    autogluon_bucket = f"s3://{notebook_artifact_bucket}/{notebook_autogluon_prefix}"
    !aws s3 sync $autogluon_bucket ./

.. code:: ipython3

    !mkdir model-training/lib
    !tar -zxvf autogluon.tar.gz -C model-training/lib --strip-components=1 --no-same-owner

Save paths for dependency requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    !cd model-training/lib && ls > ../requirements.txt
    !cd model-training && sed -i -e 's#^#lib/#' requirements.txt

Split the sample dataset into a training dataset and a test dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from sklearn.model_selection import train_test_split

    sample_df_ag = sample_df[["First5","text","returned"]]
    train_data, test_data = train_test_split(sample_df_ag, test_size=0.2, random_state=123)

.. code:: ipython3

    import sagemaker
    session = sagemaker.Session()
    bucket = session.default_bucket()

    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)

    train_s3_path = session.upload_data('train_data.csv', bucket=bucket, key_prefix='ppp_model/data')
    test_s3_path = session.upload_data('test_data.csv', bucket=bucket, key_prefix='ppp_model/data')

Run a SageMaker training job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training job takes around 20 minutes with the sample dataset. If you
want to train a model with your own data, you might need to update the
training script ``train.py`` in the\ ``model-training`` folder. If you
want to use GPU instance to achieve a better accuracy, replace
``train_instance_type`` with the desired GPU instance, and uncomment
``fit_args`` and ``hyperparameters`` to pass in the related arguments to
the training script as hyperparameters.

.. code:: ipython3

    from sagemaker.mxnet import MXNet

    # Define required label and additional parameters for Autogluon TabularPredictor
    init_args = {
      'label': 'returned'
    }

    # Define parameters for Autogluon TabularPredictor fit method
    #fit_args = {
    #  'ag_args_fit': {'num_gpus': 1}
    #}

    hyperparameters = {'init_args': str(init_args)}
    # hyperparameters = {'init_args': str(init_args), 'fit_args': str(fit_args)}

    tags = [{'Key' : 'AlgorithmName', 'Value' : 'AutoGluon-Tabular'},
            {'Key' : 'ProjectName', 'Value' : 'Jumpstart-Industry'},]

    estimator = MXNet(
        entry_point="train.py",
        role=sagemaker.get_execution_role(),
        train_instance_count=1,
        train_instance_type="ml.c5.2xlarge",
        framework_version="1.8.0",
        py_version="py37",
        source_dir="model-training",
        base_job_name='jumpstart-industry-example-ppp',
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

Download the following files (training job artifacts) from the SageMaker
session’s default S3 bucket: \* ``leaderboard.csv`` \*
``predictions.csv`` \* ``feature_importance.csv`` \* ``evaluation.json``

.. code:: ipython3

    import boto3

    s3_client = boto3.client("s3")
    job_name = estimator._current_job_name
    s3_client.download_file(bucket, f"{job_name}/output/output.tar.gz", "output.tar.gz")
    !tar -xvzf output.tar.gz

The result of the training evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    import json

    with open('evaluation.json') as f:
        data = json.load(f)
    print(data)

The ``evaluation.json`` file reports all the usual metrics as well as
the Matthews correlation coefficient (MCC). This is a more comprehensive
metric for an unbalanced dataset. It ranges from :math:`-1` to
:math:`+1`, where :math:`-1` implies perfect misclassification and
:math:`+1` is perfect classification.

   **Reference**: Davide Chicco & Giuseppe Jurman (2020), `The
   advantages of the Matthews correlation coefficient (MCC) over F1
   score and accuracy in binary classification
   evaluation <https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7>`__,
   BMC Genomics volume 21, Article number: 6

   **Note**: Various metrics are discussed in `Receiver operating
   characteristic <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`__
   in Wikipedia, the free encyclopedia.

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
