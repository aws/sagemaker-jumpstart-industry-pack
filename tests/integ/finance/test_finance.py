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
from __future__ import print_function, absolute_import

import os
import pandas as pd
import boto3
import io

from smjumpstart.finance.nlp_score_type import NLPScoreType
from smjumpstart.finance.processor import Summarizer, NLPScorer, DataLoader, SECXMLFilingParser
from smjumpstart.finance.processor_config import (
    NLPScorerConfig,
    JaccardSummarizerConfig,
    KMedoidsSummarizerConfig,
    EDGARDataSetConfig,
)
from smjumpstart.finance.nlp_score_type import NO_WORD_LIST
from tests.integ import DATA_DIR, timeout, utils


FINANCE_DEFAULT_TIMEOUT_MINUTES = 15


def test_jaccard_summarizer(
    sagemaker_session,
    cpu_instance_type,
):
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            jaccard_summarizer_config = JaccardSummarizerConfig(summary_size=100)
            data_path = os.path.join(DATA_DIR, "finance", "processor_data.csv")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            prefix = "{}/{}".format("jumpstart-gecko-jaccard-summarizer", test_run)
            s3_output_path = "s3://{}/{}".format(bucket, prefix)
            output_file_name = "output.csv"
            jaccard_summarizer = Summarizer(
                role="SageMakerRole",
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            jaccard_summarizer.summarize(
                jaccard_summarizer_config,
                "text",
                data_path,
                s3_output_path,
                output_file_name,
                new_summary_column_name="summary",
            )
        check_output_file_exists(
            jaccard_summarizer, sagemaker_session, bucket, prefix, output_file_name
        )
        check_output_file_new_columns(
            jaccard_summarizer, bucket, prefix, output_file_name, ["summary"]
        )
    finally:
        remove_test_resources(jaccard_summarizer, bucket, prefix)


def test_kmedoids_summarizer(
    sagemaker_session,
    cpu_instance_type,
):
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            kmedoids_summarizer_config = KMedoidsSummarizerConfig(100)
            data_path = os.path.join(DATA_DIR, "finance", "processor_data.csv")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            prefix = "{}/{}".format("jumpstart-gecko-kmedoids-summarizer", test_run)
            s3_output_path = "s3://{}/{}".format(bucket, prefix)
            output_file_name = "output.csv"
            kmedoids_summarizer = Summarizer(
                role="SageMakerRole",
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            kmedoids_summarizer.summarize(
                kmedoids_summarizer_config,
                "text",
                data_path,
                s3_output_path,
                output_file_name,
                new_summary_column_name="summary",
            )
        check_output_file_exists(
            kmedoids_summarizer, sagemaker_session, bucket, prefix, output_file_name
        )
        check_output_file_new_columns(
            kmedoids_summarizer, bucket, prefix, output_file_name, ["summary"]
        )
    finally:
        remove_test_resources(kmedoids_summarizer, bucket, prefix)


def test_nlp_scorer(
    sagemaker_session,
    cpu_instance_type,
):
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            score_type_list = list(
                NLPScoreType(score_type, [])
                for score_type in NLPScoreType.DEFAULT_SCORE_TYPES
                if score_type not in NO_WORD_LIST
            )
            score_type_list.extend([NLPScoreType(score_type, None) for score_type in NO_WORD_LIST])
            nlp_scorer_config = NLPScorerConfig(score_type_list)
            data_path = os.path.join(DATA_DIR, "finance", "processor_data.csv")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            prefix = "{}/{}".format("jumpstart-gecko-nlp-scorer", test_run)
            s3_output_path = "s3://{}/{}".format(bucket, prefix)
            output_file_name = "output.csv"
            nlp_scorer = NLPScorer(
                role="SageMakerRole",
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            nlp_scorer.calculate(
                nlp_scorer_config, "text", data_path, s3_output_path, output_file_name
            )
        check_output_file_exists(nlp_scorer, sagemaker_session, bucket, prefix, output_file_name)
        check_output_file_new_columns(
            nlp_scorer, bucket, prefix, output_file_name, list(NLPScoreType.DEFAULT_SCORE_TYPES)
        )
    finally:
        remove_test_resources(nlp_scorer, bucket, prefix)


def test_dataloader(
    sagemaker_session,
    cpu_instance_type,
):
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            dataset_config = EDGARDataSetConfig(
                tickers_or_ciks=["amzn"],
                form_types=["10-Q"],
                filing_date_start="2020-01-01",
                filing_date_end="2020-03-31",
                email_as_user_agent="text@xyz.com",
            )
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            prefix = "{}/{}".format("jumpstart-gecko-sec-filing-retrieval", test_run)
            s3_output_path = "s3://{}/{}".format(bucket, prefix)
            output_file_name = "output.csv"
            dataloader = DataLoader(
                role="SageMakerRole",
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            dataloader.load(
                dataset_config,
                s3_output_path,
                output_file_name,
            )
        check_output_file_exists(dataloader, sagemaker_session, bucket, prefix, output_file_name)
        check_output_file_new_columns(
            dataloader,
            bucket,
            prefix,
            output_file_name,
            ["ticker", "form_type", "accession_number", "filing_date", "text"],
        )
    finally:
        remove_test_resources(dataloader, bucket, prefix)


def test_sec_xml_filing_parser(
    sagemaker_session,
    cpu_instance_type,
):
    try:
        with timeout.timeout(minutes=FINANCE_DEFAULT_TIMEOUT_MINUTES):
            input_data_folder = os.path.join(DATA_DIR, "finance", "sec_filings")
            test_run = utils.unique_name_from_base("test_run")
            bucket = sagemaker_session.default_bucket()
            prefix = "{}/{}".format("jumpstart-gecko-sec-parser", test_run)
            s3_output_path = "s3://{}/{}".format(bucket, prefix)
            parser = SECXMLFilingParser(
                role="SageMakerRole",
                instance_count=1,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
            )
            parser.parse(
                input_data_folder,
                s3_output_path,
            )
        check_output_file_exists(parser, sagemaker_session, bucket, prefix, "parsed")
    finally:
        remove_test_resources(parser, bucket, prefix)


def check_output_file_exists(processor, sagemaker_session, bucket, prefix, output_file_name):
    if processing_job_completed(sagemaker_session.sagemaker_client, processor._current_job_name):
        s3_client = boto3.client("s3", region_name=processor.sagemaker_session.boto_region_name)
        response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
        assert output_file_name in response["Contents"][0]["Key"]


def check_output_file_new_columns(processor, bucket, prefix, output_file_name, new_column_names):
    s3_client = boto3.client("s3", region_name=processor.sagemaker_session.boto_region_name)
    output_file_object = s3_client.get_object(Bucket=bucket, Key=prefix + "/" + output_file_name)
    output_df = pd.read_csv(io.BytesIO(output_file_object["Body"].read()))
    assert all(col_name in output_df.columns for col_name in new_column_names)


def processing_job_completed(sagemaker_client, job_name):
    response = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
    if not response or "ProcessingJobStatus" not in response:
        raise ValueError("Response is none or does not have ProcessingJobStatus")
    status = response["ProcessingJobStatus"]
    return status == "Completed"


def remove_test_resources(processor, bucket, prefix):
    s3_resource = boto3.resource("s3")
    bucket_obj = s3_resource.Bucket(bucket)
    bucket_obj.objects.filter(Prefix=prefix).delete()
    processing_job_folder = processor._current_job_name
    bucket_obj.objects.filter(Prefix=processing_job_folder).delete()
