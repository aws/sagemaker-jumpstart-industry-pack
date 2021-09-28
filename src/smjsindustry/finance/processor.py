# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""The SageMaker JumpStart Industry processing job module.

These classes assist in the automatic creation of SageMaker
processing jobs that perform heavy-duty computational tasks that
are useful in financial use cases. Such processing jobs include
but are not limited to downloading and parsing SEC filings from
the EDGAR database, summarizing text using the Jaccard or k-medoids
algorithms, and scoring documents using NLP techniques.

"""
from __future__ import print_function, absolute_import

import json
import logging
import os
import tempfile
import copy
from typing import Dict, List, Union
from six.moves.urllib.parse import urlparse

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from smjsindustry.finance.processor_config import (
    JaccardSummarizerConfig,
    KMedoidsSummarizerConfig,
    NLPScorerConfig,
    EDGARDataSetConfig,
)
from smjsindustry.finance.constants import (
    SUMMARIZER_JOB_NAME,
    NLP_SCORE_JOB_NAME,
    JACCARD_SUMMARIZER,
    SEC_FILING_PARSER_JOB_NAME,
    SEC_XML_FILING_PARSER,
    SEC_FILING_RETRIEVAL_JOB_NAME,
)
from smjsindustry.finance.utils import retrieve_image

logger = logging.getLogger()


class FinanceProcessor(Processor):
    """Handles SageMaker JumpStart Industry processing tasks.

    This base class is for handling SageMaker JumpStart Industry processing tasks.
    See its subclasses, such as :class:`~smjsindustry.Summarizer`
    and :class:`~smjsindustry.NLPScorer`, for concrete
    examples of ``FinanceProcessors`` that perform specific computation tasks.

    Args:
        role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
            uses this role to access AWS resources, such as
            data stored in Amazon S3.
        instance_count (int): The number of instances to run
            a processing job with.
        instance_type (str): The type of Amazon EC2 instance to use for
            processing. For example, ``'ml.c4.xlarge'``.
        volume_size_in_gb (int): Size in GB of the EBS volume
            to use for storing data during processing (default: 30).
        volume_kms_key (str): An AWS KMS key for the processing
            volume (default: None).
        output_kms_key (str): The AWS KMS key ID for processing job outputs (default: None).
        max_runtime_in_seconds (int): Timeout in seconds (default: None).
            After this amount of time, Amazon SageMaker terminates the job,
            regardless of its current status. If ``max_runtime_in_seconds`` is not
            specified, the default value is 24 hours.
        sagemaker_session (:class:`~sagemaker.session.Session`):
            A `SageMaker Session
            <https://sagemaker.readthedocs.io/en/stable/api/utility/session.html#sagemaker.session.Session>`_
            object which manages interactions with Amazon SageMaker and
            any other AWS services needed. If not specified, the processor creates
            one using the default AWS configuration chain.
        tags (List[Dict[str, str]]): List of tags to be passed to the processing job
            (default: None). To learn more more, see
            `Tag <https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html>`_
            in the *Amazon SageMaker API Reference*.
        base_job_name (str):
            A prefix for the processing job name. If not specified,
            the processor generates a default job name,
            based on the processing image name and the current timestamp.
        network_config (:class:`~sagemaker.network.NetworkConfig`):
            A `SageMaker NatworkConfig <https://sagemaker.readthedocs.io/en/stable/api/utility/network.html>`_
            object that configures network isolation, encryption of
            inter-container traffic, security group IDs, and subnets.

    """

    _PROCESSING_CONFIG = "/opt/ml/processing/input/config"
    _PROCESSING_DATA = "/opt/ml/processing/input/data"
    _PROCESSING_OUTPUT = "/opt/ml/processing/output"
    _CONFIG_FILE = "job_config.json"
    _CONFIG_INPUT_NAME = "config"
    _DATA_INPUT_NAME = "data"

    def __init__(
        self,
        role: str,
        instance_count: int,
        instance_type: str,
        volume_size_in_gb: int = 30,
        volume_kms_key: str = None,
        output_kms_key: str = None,
        max_runtime_in_seconds: int = None,
        sagemaker_session: sagemaker.session.Session = None,
        tags: List[Dict[str, str]] = None,
        base_job_name: str = None,
        network_config: sagemaker.network.NetworkConfig = None,
    ):
        """Initializes a ``Processor`` instance for SageMaker JumpStart Industry processing jobs."""
        container_uri = retrieve_image(sagemaker_session.boto_region_name)
        super(FinanceProcessor, self).__init__(
            role,
            container_uri,
            instance_count,
            instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            tags=tags,
            base_job_name=base_job_name,
            network_config=network_config,
        )

    def run(self, **kwargs):
        """Overrides the base class method."""
        logger.info("You are not charged when EC2 instances are in pending state")
        logger.info(
            "More info: "
            "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html"
        )
        super(FinanceProcessor, self).run(**kwargs)


class Summarizer(FinanceProcessor):
    """Initializes a Summarizer instance that summarizes text.

    For the general processing job configuration parameters of this class,
    see the parameters in the
    :class:`~smjsindustry.finance.processor.FinanceProcessor` class.

    It summarizes text while preserving key information content and overall meaning.
    Summarization can be performed using either the Jaccard algorithm or the k-medoids algorithm.
    See the summarize methods for details regarding the specific algorithms used.

    """

    def __init__(
        self,
        role: str,
        instance_count: int,
        instance_type: str,
        volume_size_in_gb: int = 30,
        volume_kms_key: str = None,
        output_kms_key: str = None,
        max_runtime_in_seconds: int = None,
        sagemaker_session: sagemaker.session.Session = None,
        tags: List[Dict[str, str]] = None,
        network_config: sagemaker.network.NetworkConfig = None,
    ):
        """Initializes a Summarizer instance to summarize text.

        The Summarizer instance handles text summarization to provide a concise summary
        while preserving key information content and overall meaning. Please see
        the summarize method for details regarding the specific algorithms used.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            instance_count (int): The number of instances to run
                a processing job with.
            instance_type (str): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume (default: None).
            output_kms_key (str): The KMS key ID for processing job outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            tags (List[Dict[str, str]]): List of tags to be passed to the processing job
                (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        super(Summarizer, self).__init__(
            role,
            instance_count,
            instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            tags=tags,
            base_job_name=SUMMARIZER_JOB_NAME,
            network_config=network_config,
        )

    def summarize(
        self,
        summarizer_config: Union[JaccardSummarizerConfig, KMedoidsSummarizerConfig],
        text_column_name: str,
        input_file_path: str,
        s3_output_path: str,
        output_file_name: str,
        new_summary_column_name: str = "summary",
        wait: bool = True,
        logs: bool = True,
    ):
        """Runs a processing job to generate Jaccard or k-medoid summary.

        The summaries generated by the Jaccard algorithm give the main theme of the document
        by extracting the sentences with the greatest similarity among all sentences.
        Similarity is measured using the Jaccard coefficient, which, for a pair of sentences,
        is the number of common words between them normalized by the size of the super set
        of the words in the two sentences.

        The k-medoids algorithm clusters sentences and
        outputs the medoids of each cluster as a summary.

        Args:
            summarizer_config (Union[JaccardSummarizerConfig, KMedoidsSummarizerConfig]):
                The config for the JaccardSummarizer or KmedoidSummarizer.
            text_column_name (str): The name for column containing text to be summarized.
            input_file_path (str): The input file path pointing to the input dataframe
                containing the text to be summarized. It can be a local file or a S3 path.
            s3_output_path (str): An S3 prefix in the format of
                ``'s3://<output bucket name>/output/path'``.
            output_file_name (str): The output file name. The full path is
                ``'s3://<output bucket name>/output/path/output_file_name'``.
            new_summary_column_name (str): The column name for the summary in the given
                dataframe (default: ``"summary"``).
            wait (bool): Whether the call should wait until the job completes (default: ``True``).
            logs (bool): Whether to show the logs produced by the job (default: ``True``).

        Raises:
            ValueError: if ``logs`` is True but ``wait`` is False.
        """
        parse_result = urlparse(s3_output_path)
        if parse_result.scheme != "s3":
            raise Exception(
                "Expected an S3 prefix in the format of s3://<output bucket name>/output/path"
            )
        with tempfile.TemporaryDirectory() as tmpdirname:
            summarizer_config_file = os.path.join(tmpdirname, self._CONFIG_FILE)
            with open(summarizer_config_file, "w") as file_handle:
                cloned_config = copy.deepcopy(summarizer_config.get_config())
                if cloned_config["processor_type"] == JACCARD_SUMMARIZER:
                    if isinstance(cloned_config["vocabulary"], set):
                        cloned_config["vocabulary"] = list(cloned_config["vocabulary"])
                cloned_config["text_column_name"] = text_column_name
                cloned_config["new_summary_column_name"] = new_summary_column_name
                cloned_config["output_file_name"] = output_file_name
                json.dump(cloned_config, file_handle)
            config_input = ProcessingInput(
                source=tmpdirname,
                destination=self._PROCESSING_CONFIG,
                input_name=self._CONFIG_INPUT_NAME,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
            data_input = ProcessingInput(
                source=input_file_path,
                destination=self._PROCESSING_DATA,
                input_name=self._DATA_INPUT_NAME,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
            result_output = ProcessingOutput(
                source=self._PROCESSING_OUTPUT,
                destination=s3_output_path,
                s3_upload_mode="EndOfJob",
            )
            logger.info("Starting SageMaker processing job to summarize")
            super().run(
                inputs=[config_input, data_input],
                outputs=[result_output],
                wait=wait,
                logs=logs,
            )
            logger.info("Completed SageMaker processing job to summarize")


class NLPScorer(FinanceProcessor):
    """Calculates NLP scores for text using default or user-provided word lists.

    Text that contains many words and phrases that are related to the provided
    word lists will receive high scores while text that is unrelated will score lower.

    The NLP scores report the percentage of words in a document that match
    a list of words, which is called lexicon.
    The matching is undertaken after stemming of the document and the lexicon.
    NLP scoring of sentiment is based on the Vader sentiment lexicon.
    NLP Scoring of readability is based on the Gunning-Fog index.

    For the general processing job configuration parameters of this class,
    see the parameters in the
    :class:`~smjsindustry.finance.processor.FinanceProcessor` class.

    """

    def __init__(
        self,
        role: str,
        instance_count: int,
        instance_type: str,
        volume_size_in_gb: int = 30,
        volume_kms_key: str = None,
        output_kms_key: str = None,
        max_runtime_in_seconds: int = None,
        sagemaker_session: sagemaker.session.Session = None,
        tags: List[Dict[str, str]] = None,
        network_config: sagemaker.network.NetworkConfig = None,
    ):
        """Initializes an NLPScorer instance to calculate NLP scores for text.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            instance_count (int): The number of instances to run
                a processing job with.
            instance_type (str): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume (default: None).
            output_kms_key (str): The KMS key ID for processing job outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            tags (List[Dict[str, str]]): List of tags to be passed to the processing job
                (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        super(NLPScorer, self).__init__(
            role,
            instance_count,
            instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            tags=tags,
            base_job_name=NLP_SCORE_JOB_NAME,
            network_config=network_config,
        )

    def calculate(
        self,
        score_config: NLPScorerConfig,
        text_column_name: str,
        input_file_path: str,
        s3_output_path: str,
        output_file_name: str,
        wait: bool = True,
        logs: bool = True,
    ):
        """Runs a processing job to generate NLP scores for input text.

        Args:
            score_config (:class:`~smjsindustry.NLPScorerConfig`):
                The config for the NLP scorer.
            text_column_name (str): The name for column containing text to be summarized.
            input_file_path (str): The input file path pointing to the input dataframe
                containing the text to be summarized. It can be a local path or a S3 path.
            s3_output_path (str): An S3 prefix in the format of
                ``'s3://<output bucket name>/output/path'``.
            output_file_name (str): The output file name. The full path is
                ``'s3://<output bucket name>/output/path/output_file_name'``.
            wait (bool): Whether the call should wait until the job completes (default: ``True``).
            logs (bool): Whether to show the logs produced by the job (default: ``True``).

        Raises:
            ValueError: if ``logs`` is True but ``wait`` is False.

        """
        parse_result = urlparse(s3_output_path)
        if parse_result.scheme != "s3":
            raise Exception(
                "Expected an S3 prefix in the format of s3://<output bucket name>/output/path"
            )
        with tempfile.TemporaryDirectory() as tmpdirname:
            score_config_file = os.path.join(tmpdirname, self._CONFIG_FILE)
            with open(score_config_file, "w") as file_handle:
                cloned_config = copy.deepcopy(score_config.get_config())
                cloned_config["text_column_name"] = text_column_name
                cloned_config["output_file_name"] = output_file_name
                json.dump(cloned_config, file_handle)
            config_input = ProcessingInput(
                source=tmpdirname,
                destination=self._PROCESSING_CONFIG,
                input_name=self._CONFIG_INPUT_NAME,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
            data_input = ProcessingInput(
                source=input_file_path,
                destination=self._PROCESSING_DATA,
                input_name=self._DATA_INPUT_NAME,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
            result_output = ProcessingOutput(
                source=self._PROCESSING_OUTPUT,
                destination=s3_output_path,
                s3_upload_mode="EndOfJob",
            )
            logger.info("Starting SageMaker processing job to calculate NLP scores")
            super().run(
                inputs=[config_input, data_input],
                outputs=[result_output],
                wait=wait,
                logs=logs,
            )
            logger.info("Completed SageMaker processing job to calculate NLP scores")


class DataLoader(FinanceProcessor):
    """Initializes a DataLoader instance to load a dataset.

    For the general processing job configuration parameters of this class,
    see the parameters in the
    :class:`~smjsindustry.finance.processor.FinanceProcessor` class.

    The following ``load`` class method with
    :class:`~smjsindustry.finance.EDGARDataSetConfig`
    downloads SEC XML filings from the `SEC EDGAR database <https://www.sec.gov/edgar/>`_
    and parses the downloaded XML filings to plain text files.

    """

    def __init__(
        self,
        role: str,
        instance_count: int,
        instance_type: str,
        volume_size_in_gb: int = 30,
        volume_kms_key: str = None,
        output_kms_key: str = None,
        max_runtime_in_seconds: int = None,
        sagemaker_session: sagemaker.session.Session = None,
        tags: List[Dict[str, str]] = None,
        network_config: sagemaker.network.NetworkConfig = None,
    ):
        """Initializes a DataLoader instance to load data from the `SEC EDGAR database <https://www.sec.gov/edgar/>`_.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            instance_count (int): The number of instances to run
                a processing job with.
            instance_type (str): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume (default: None).
            output_kms_key (str): The KMS key ID for processing job outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            tags (List[Dict[str, str]]): List of tags to be passed to the processing job
                (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        if instance_count > 1:
            logger.info("Dataloader processing jobs only support 1 instance.")
            instance_count = 1

        super(DataLoader, self).__init__(
            role,
            instance_count,
            instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            tags=tags,
            base_job_name=SEC_FILING_RETRIEVAL_JOB_NAME,
            network_config=network_config,
        )

    def load(
        self,
        dataset_config: EDGARDataSetConfig,
        s3_output_path: str,
        output_file_name: str,
        wait: bool = True,
        logs: bool = True,
    ):
        """Runs a processing job to load dataset from `SEC EDGAR database <https://www.sec.gov/edgar/>`_.

        Args:
            dataset_config (:class:`~smjsindustry.finance.EDGARDataSetConfig`):
                The config for the DataLoader.
            s3_output_path (str): An S3 prefix in the format of
                ``'s3://<output bucket name>/output/path'``.
            output_file_name (str): The output file name. The full path is
                ``'s3://<output bucket name>/output/path/output_file_name'``.
            wait (bool): Whether the call should wait until the job completes (default: ``True``).
            logs (bool): Whether to show the logs produced by the job (default: ``True``).

        Raises:
            ValueError: if ``logs`` is True but ``wait`` is False.
        """
        parse_result = urlparse(s3_output_path)
        if parse_result.scheme != "s3":
            raise Exception(
                "Expected an S3 prefix in the format of s3://<output bucket name>/output/path"
            )
        with tempfile.TemporaryDirectory() as tmpdirname:
            dataset_config_file = os.path.join(tmpdirname, self._CONFIG_FILE)
            with open(dataset_config_file, "w") as file_handle:
                cloned_config = copy.deepcopy(dataset_config.get_config())
                cloned_config["output_file_name"] = output_file_name
                json.dump(cloned_config, file_handle)
            config_input = ProcessingInput(
                input_name=self._CONFIG_INPUT_NAME,
                source=tmpdirname,
                destination=self._PROCESSING_CONFIG,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
            result_output = ProcessingOutput(
                source=self._PROCESSING_OUTPUT,
                destination=s3_output_path,
                s3_upload_mode="EndOfJob",
            )
            logger.info("Starting SageMaker processing job to load dataset")
            super().run(
                inputs=[config_input],
                outputs=[result_output],
                wait=wait,
                logs=logs,
            )
            logger.info("Completed SageMaker processing job to load dataset")


class SECXMLFilingParser(FinanceProcessor):
    """Initializes a SECXMLFilingParser instance that parses SEC XML filings.

    For the general processing job configuration parameters of this class,
    see the parameters in the
    :class:`~smjsindustry.finance.processor.FinanceProcessor` class.

    The following ``parse`` class method parses user-downloaded SEC XML filings
    to plain text files.

    """

    def __init__(
        self,
        role: str,
        instance_count: int,
        instance_type: str,
        volume_size_in_gb: int = 30,
        volume_kms_key: str = None,
        output_kms_key: str = None,
        max_runtime_in_seconds: int = None,
        sagemaker_session: sagemaker.session.Session = None,
        tags: List[Dict[str, str]] = None,
        network_config: sagemaker.network.NetworkConfig = None,
    ):
        """Initializes a SECXMLFilingParser instance to parse the SEC XML filings.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            instance_count (int): The number of instances to run
                a processing job with.
            instance_type (str): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume (default: None).
            output_kms_key (str): The KMS key ID for processing job outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            tags (List[Dict[str, str]]): List of tags to be passed to the processing job
                (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        super(SECXMLFilingParser, self).__init__(
            role,
            instance_count,
            instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            tags=tags,
            base_job_name=SEC_FILING_PARSER_JOB_NAME,
            network_config=network_config,
        )

    def parse(
        self,
        input_data_path: str,
        s3_output_path: str,
        wait: bool = True,
        logs: bool = True,
    ):
        """Runs a processing job to parse SEC XML filings.

        Args:
            input_data_path (str): The input file path pointing to directory containing
                the SEC XML filings to be parsed. It can be a local folder or an S3 path.
            s3_output_path (str): An S3 prefix in the format of
                ``'s3://<output bucket name>/output/path'``.
            wait (bool): Whether the call should wait until the job completes (default: ``True``).
            logs (bool): Whether to show the logs produced by the job (default: ``True``).

        Raises:
            ValueError: if ``logs`` is True but ``wait`` is False.
        """
        parse_result = urlparse(s3_output_path)
        if parse_result.scheme != "s3":
            raise Exception(
                "Expected an S3 prefix in the format of s3://<output bucket name>/output/path"
            )
        with tempfile.TemporaryDirectory() as tmpdirname:
            parser_config_file = os.path.join(tmpdirname, self._CONFIG_FILE)
            with open(parser_config_file, "w") as file_handle:
                parser_config = {"processor_type": SEC_XML_FILING_PARSER}
                json.dump(parser_config, file_handle)
            config_input = ProcessingInput(
                source=tmpdirname,
                destination=self._PROCESSING_CONFIG,
                input_name=self._CONFIG_INPUT_NAME,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
            data_input = ProcessingInput(
                source=input_data_path,
                destination=self._PROCESSING_DATA,
                input_name=self._DATA_INPUT_NAME,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
            result_output = ProcessingOutput(
                source=self._PROCESSING_OUTPUT,
                destination=s3_output_path,
                s3_upload_mode="EndOfJob",
            )
            logger.info("Starting SageMaker processing job to parse sec filings")
            super().run(
                inputs=[config_input, data_input],
                outputs=[result_output],
                wait=wait,
                logs=logs,
            )
            logger.info("Completed SageMaker processing job to parse sec filings")
