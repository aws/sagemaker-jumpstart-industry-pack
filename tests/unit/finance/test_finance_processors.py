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
from __future__ import absolute_import

from mock import Mock, MagicMock
import pytest

from smjsindustry import (
    Summarizer,
    NLPScorer,
    JaccardSummarizerConfig,
    KMedoidsSummarizerConfig,
    NLPScorerConfig,
    NLPSCORE_NO_WORD_LIST,
    NLPScoreType,
)
from smjsindustry.finance import DataLoader, SECXMLFilingParser, EDGARDataSetConfig
from smjsindustry.finance.constants import (
    JACCARD_SUMMARIZER,
    KMEDOIDS_SUMMARIZER,
    SUMMARIZER_JOB_NAME,
    NLP_SCORER,
    NLP_SCORE_JOB_NAME,
    LOAD_DATA,
    SEC_FILING_RETRIEVAL_JOB_NAME,
    SEC_FILING_PARSER_JOB_NAME,
    REPOSITORY,
    CONTAINER_IMAGE_VERSION,
)
from smjsindustry.finance.utils import retrieve_image

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
ROLE = "arn:aws:iam::627189473827:role/SageMakerRole"
IMAGE_URI = "935494966801.dkr.ecr.us-west-2.amazonaws.com/{}:{}".format(
    REPOSITORY, CONTAINER_IMAGE_VERSION
)


@pytest.fixture(scope="module")
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)

    session_mock.upload_data = Mock(
        name="upload_data", return_value="mocked_s3_uri_from_upload_data"
    )
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = ROLE
    session_mock.describe_processing_job = MagicMock(
        name="describe_processing_job", return_value=get_describe_response_inputs_and_outputs()
    )
    return session_mock


@pytest.fixture(scope="module")
def jaccard_summarizer_config():
    return JaccardSummarizerConfig(summary_size=100, vocabulary=set(["bad", "distress", "angry"]))


@pytest.fixture(scope="module")
def kmedoids_summarizer_config():
    return KMedoidsSummarizerConfig(50)


@pytest.fixture(scope="module")
def nlp_scorer_config():
    nlp_score_types = [
        NLPScoreType(NLPScoreType.POSITIVE, ["good", "great", "happy"]),
        NLPScoreType(NLPScoreType.NEGATIVE, ["bad", "unhappy", "sad", "terrible"]),
        NLPScoreType("talent", ["ability", "exceptional", "prospect", "adept"]),
    ]
    return NLPScorerConfig(nlp_score_types)


@pytest.fixture(scope="module")
def summarizer_processor(sagemaker_session):
    return Summarizer(
        role=ROLE,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture(scope="module")
def nlp_scorer(sagemaker_session):
    return NLPScorer(
        role=ROLE,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture(scope="module")
def dataset_config():
    return KMedoidsSummarizerConfig(50)


@pytest.fixture(scope="module")
def dataloader_processor(sagemaker_session):
    return DataLoader(
        role=ROLE,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture(scope="module")
def parser_processor(sagemaker_session):
    return SECXMLFilingParser(
        role=ROLE,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )


def test_image_uri():
    uri = retrieve_image("us-west-2")
    assert uri == IMAGE_URI


@pytest.mark.parametrize(
    "config_arg", ["summary_size", "summary_percentage", "max_tokens", "cutoff", "vocabulary"]
)
@pytest.mark.parametrize("vocabulary", [None, set(["good", "great", "nice"])])
def test_jaccard_summarizer_config(config_arg, vocabulary):
    with pytest.raises(ValueError) as error:
        JaccardSummarizerConfig(summary_size=0, summary_percentage=0, vocabulary=vocabulary)
        JaccardSummarizerConfig(vocabulary=vocabulary)
    assert (
        "Only one summary size related argument can be specified, "
        "choose to specify one from summary_size, summary_percentage, max_tokens, cutoff."
        in str(error.value)
    )
    if config_arg == "summary_size":
        summarizer_config = JaccardSummarizerConfig(summary_size=100, vocabulary=vocabulary)
        expected_config = {
            "processor_type": JACCARD_SUMMARIZER,
            "summary_size": 100,
            "summary_percentage": 0,
            "max_tokens": 0,
            "cutoff": 0,
            "vocabulary": vocabulary,
        }
        assert summarizer_config.get_config() == expected_config
    if config_arg == "summary_percentage":
        summarizer_config = JaccardSummarizerConfig(summary_percentage=0.2, vocabulary=vocabulary)
        expected_config = {
            "processor_type": JACCARD_SUMMARIZER,
            "summary_size": 0,
            "summary_percentage": 0.2,
            "max_tokens": 0,
            "cutoff": 0,
            "vocabulary": vocabulary,
        }
        assert summarizer_config.get_config() == expected_config
    if config_arg == "max_tokens":
        summarizer_config = JaccardSummarizerConfig(max_tokens=1000, vocabulary=vocabulary)
        expected_config = {
            "processor_type": JACCARD_SUMMARIZER,
            "summary_size": 0,
            "summary_percentage": 0,
            "max_tokens": 1000,
            "cutoff": 0,
            "vocabulary": vocabulary,
        }
        assert summarizer_config.get_config() == expected_config
    if config_arg == "cutoff":
        summarizer_config = JaccardSummarizerConfig(cutoff=0.5, vocabulary=vocabulary)
        expected_config = {
            "processor_type": JACCARD_SUMMARIZER,
            "summary_size": 0,
            "summary_percentage": 0,
            "max_tokens": 0,
            "cutoff": 0.5,
            "vocabulary": vocabulary,
        }
        assert summarizer_config.get_config() == expected_config


@pytest.mark.parametrize("config_arg", ["vector_size", "min_count", "epochs", "metric", "init"])
def test_kmedoids_summarizer_config(config_arg):
    if config_arg == "vector_size":
        summarizer_config = KMedoidsSummarizerConfig(100, vector_size=50)
        expected_config = {
            "processor_type": KMEDOIDS_SUMMARIZER,
            "summary_size": 100,
            "vector_size": 50,
            "min_count": 0,
            "epochs": 60,
            "metric": "euclidean",
            "init": "heuristic",
        }
        assert summarizer_config.get_config() == expected_config
    if config_arg == "min_count":
        summarizer_config = KMedoidsSummarizerConfig(80, min_count=10)
        expected_config = {
            "processor_type": KMEDOIDS_SUMMARIZER,
            "summary_size": 80,
            "vector_size": 100,
            "min_count": 10,
            "epochs": 60,
            "metric": "euclidean",
            "init": "heuristic",
        }
        assert summarizer_config.get_config() == expected_config
    if config_arg == "epochs":
        summarizer_config = KMedoidsSummarizerConfig(50, epochs=10)
        expected_config = {
            "processor_type": KMEDOIDS_SUMMARIZER,
            "summary_size": 50,
            "vector_size": 100,
            "min_count": 0,
            "epochs": 10,
            "metric": "euclidean",
            "init": "heuristic",
        }
        assert summarizer_config.get_config() == expected_config
    if config_arg == "metric":
        summarizer_config = KMedoidsSummarizerConfig(150, metric="cosine")
        expected_config = {
            "processor_type": KMEDOIDS_SUMMARIZER,
            "summary_size": 150,
            "vector_size": 100,
            "min_count": 0,
            "epochs": 60,
            "metric": "cosine",
            "init": "heuristic",
        }
        assert summarizer_config.get_config() == expected_config
    if config_arg == "init":
        summarizer_config = KMedoidsSummarizerConfig(200, init="random")
        expected_config = {
            "processor_type": KMEDOIDS_SUMMARIZER,
            "summary_size": 200,
            "vector_size": 100,
            "min_count": 0,
            "epochs": 60,
            "metric": "euclidean",
            "init": "random",
        }
        assert summarizer_config.get_config() == expected_config


@pytest.mark.parametrize(
    "score_type",
    [
        NLPScoreType(NLPScoreType.POSITIVE, []),
        NLPScoreType(NLPScoreType.POSITIVE, ["good", "great"]),
        NLPScoreType("custom", ["good", "great"]),
        NLPScoreType(NLPScoreType.POLARITY, None),
        17,
    ],
)
def test_nlp_scorer_config(score_type):
    if not isinstance(score_type, NLPScoreType):
        with pytest.raises(TypeError) as error:
            NLPScorerConfig(score_type)
            NLPScorerConfig([score_type])
        assert str(error.value) == (
            "An NLPScorerConfig must be initialized with "
            "either a single NLPScoreType object, or "
            "a list of NLPScoreType objects."
        )
    else:
        config = NLPScorerConfig(score_type).get_config()
        config_with_list_input = NLPScorerConfig([score_type]).get_config()
        expected_config = {
            "processor_type": NLP_SCORER,
            "score_types": {score_type.score_name: score_type.word_list},
        }
        assert config == expected_config
        assert config_with_list_input == expected_config


def test_jaccard_summarize(
    summarizer_processor,
    jaccard_summarizer_config,
    sagemaker_session,
):
    s3_input_path = "s3://input"
    s3_output_path = "s3://output"
    summarizer_processor.summarize(
        jaccard_summarizer_config,
        "text",
        s3_input_path,
        s3_output_path,
        "output.csv",
        new_summary_column_name="summary",
    )
    expected_args = get_expected_args_all_parameters(
        summarizer_processor._current_job_name,
        s3_output_path,
        s3_input_path,
    )
    sagemaker_session.process.assert_called_with(**expected_args)
    assert SUMMARIZER_JOB_NAME in summarizer_processor._current_job_name
    assert expected_args["app_specification"]["ImageUri"] == IMAGE_URI
    assert expected_args["inputs"][1]["S3Input"]["S3Uri"] == s3_input_path
    assert expected_args["output_config"]["Outputs"][0]["S3Output"]["S3Uri"] == s3_output_path


def test_kmedoids_summarize(
    summarizer_processor,
    kmedoids_summarizer_config,
    sagemaker_session,
):
    s3_input_path = "s3://input"
    s3_output_path = "s3://output"
    summarizer_processor.summarize(
        kmedoids_summarizer_config,
        "text",
        s3_input_path,
        s3_output_path,
        "output.csv",
        new_summary_column_name="summary",
    )
    expected_args = get_expected_args_all_parameters(
        summarizer_processor._current_job_name,
        s3_output_path,
        s3_input_path,
    )
    sagemaker_session.process.assert_called_with(**expected_args)
    assert SUMMARIZER_JOB_NAME in summarizer_processor._current_job_name
    assert expected_args["app_specification"]["ImageUri"] == IMAGE_URI
    assert expected_args["inputs"][1]["S3Input"]["S3Uri"] == s3_input_path
    assert expected_args["output_config"]["Outputs"][0]["S3Output"]["S3Uri"] == s3_output_path


def test_nlp_scorer(
    nlp_scorer,
    nlp_scorer_config,
    sagemaker_session,
):
    s3_input_path = "s3://input"
    s3_output_path = "s3://output"
    nlp_scorer.calculate(nlp_scorer_config, "text", s3_input_path, s3_output_path, "output.csv")
    expected_args = get_expected_args_all_parameters(
        nlp_scorer._current_job_name,
        s3_output_path,
        s3_input_path,
    )
    sagemaker_session.process.assert_called_with(**expected_args)
    assert NLP_SCORE_JOB_NAME in nlp_scorer._current_job_name
    assert expected_args["app_specification"]["ImageUri"] == IMAGE_URI
    assert expected_args["inputs"][1]["S3Input"]["S3Uri"] == s3_input_path
    assert expected_args["output_config"]["Outputs"][0]["S3Output"]["S3Uri"] == s3_output_path


@pytest.mark.parametrize(
    "score_name",
    [NLPScoreType.POSITIVE, "custom_positive", NLPScoreType.POLARITY, NLPScoreType.READABILITY],
)
@pytest.mark.parametrize("word_list", [["good", "great", "happy"], None, [], 17, [17, "yellow"]])
def test_nlp_score_type(score_name, word_list):
    if score_name in NLPSCORE_NO_WORD_LIST:
        if word_list is not None:
            with pytest.raises(TypeError) as error:
                NLPScoreType(score_name, word_list)
            assert str(error.value) == (
                "NLPScoreType with score_name {} requires its word_list argument to be None."
            ).format(score_name)
    else:
        if not isinstance(word_list, list):
            with pytest.raises(TypeError) as error:
                NLPScoreType(score_name, word_list)
            assert str(error.value) == (
                "NLPScoreType with score_name {} requires its word_list argument to be a list."
            ).format(score_name)
        elif score_name in NLPScoreType.DEFAULT_SCORE_TYPES:
            if word_list and any(not isinstance(word, str) for word in word_list):
                with pytest.raises(TypeError) as error:
                    NLPScoreType(score_name, word_list)
                assert str(error.value) == "word_list argument must contain only strings."
        else:
            if not word_list:
                with pytest.raises(ValueError) as error:
                    NLPScoreType(score_name, word_list)
                assert str(error.value) == (
                    "NLPScoreType with custom score_name {} requires "
                    "its word_list argument to be a non-empty list."
                ).format(score_name)
            elif any(not isinstance(word, str) for word in word_list):
                with pytest.raises(TypeError) as error:
                    NLPScoreType(score_name, word_list)
                assert str(error.value) == "word_list argument must contain only strings."
            else:
                score_type = NLPScoreType(score_name, word_list)
                assert score_type.score_name == score_name
                assert score_type.word_list == word_list


@pytest.mark.parametrize("tickers_or_ciks", [["amzn", "goog", "0000027904"], [12, 2], [], None])
@pytest.mark.parametrize("form_types", [["10-K", "10-Q"], ["invalid-form-type"], [], None])
@pytest.mark.parametrize("filing_date_start", ["2020-10-01", "1234", "", None])
@pytest.mark.parametrize("filing_date_end", ["2020-11-01", "2010-10", "", None])
@pytest.mark.parametrize("email_as_user_agent", ["user@test.com", "abc", "", None])
def test_edgar_dataset_config(
    tickers_or_ciks,
    form_types,
    filing_date_start,
    filing_date_end,
    email_as_user_agent,
):
    if tickers_or_ciks in ([12, 2], [], None):
        with pytest.raises(TypeError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert "EDGARDataSetConfig requires tickers_or_ciks to be a list of strings." in str(
            error.value
        )
    elif form_types in ([], None):
        with pytest.raises(TypeError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert "EDGARDataSetConfig requires form_types to be a list of strings." in str(error.value)
    elif form_types == ["invalid-form-type"]:
        with pytest.raises(ValueError, match=r".* not supported.$"):
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
    elif filing_date_start is None:
        with pytest.raises(TypeError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert "EDGARDataSetConfig requires filing_date_start to be a string." in str(error.value)
    elif filing_date_start in ("", "1234"):
        with pytest.raises(ValueError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert (
            "EDGARDataSetConfig requires filing_date_start in the format of 'YYYY-MM-DD'."
            in str(error.value)
        )
    elif filing_date_end is None:
        with pytest.raises(TypeError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert "EDGARDataSetConfig requires filing_date_end to be a string." in str(error.value)
    elif filing_date_end in ("", "2010-10"):
        with pytest.raises(ValueError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert "EDGARDataSetConfig requires filing_date_end in the format of 'YYYY-MM-DD'." in str(
            error.value
        )
    elif email_as_user_agent is None:
        with pytest.raises(TypeError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert "EDGARDataSetConfig requires email_as_user_agent to be a string." in str(error.value)
    elif email_as_user_agent in ("", "abc"):
        with pytest.raises(ValueError) as error:
            EDGARDataSetConfig(
                tickers_or_ciks=tickers_or_ciks,
                form_types=form_types,
                filing_date_start=filing_date_start,
                filing_date_end=filing_date_end,
                email_as_user_agent=email_as_user_agent,
            )
        assert (
            "EDGARDataSetConfig requires email_as_user_agent to be a valid email address."
            in str(error.value)
        )
    else:
        dataset_config = EDGARDataSetConfig(
            tickers_or_ciks=tickers_or_ciks,
            form_types=form_types,
            filing_date_start=filing_date_start,
            filing_date_end=filing_date_end,
            email_as_user_agent=email_as_user_agent,
        )
        expected_config = {
            "processor_type": LOAD_DATA,
            "tickers_or_ciks": tickers_or_ciks,
            "form_types": form_types,
            "filing_date_start": filing_date_start,
            "filing_date_end": filing_date_end,
            "email_as_user_agent": email_as_user_agent,
        }
        assert dataset_config.get_config() == expected_config


def test_dataloader(
    dataloader_processor,
    dataset_config,
    sagemaker_session,
):
    s3_output_path = "s3://output"
    dataloader_processor.load(
        dataset_config,
        s3_output_path,
        "output.csv",
    )
    expected_args = get_expected_args_all_parameters(
        dataloader_processor._current_job_name, s3_output_path
    )
    sagemaker_session.process.assert_called_with(**expected_args)
    assert SEC_FILING_RETRIEVAL_JOB_NAME in dataloader_processor._current_job_name
    assert expected_args["app_specification"]["ImageUri"] == IMAGE_URI
    assert expected_args["output_config"]["Outputs"][0]["S3Output"]["S3Uri"] == s3_output_path


def test_parser(
    parser_processor,
    sagemaker_session,
):
    s3_input_path = "s3://input"
    s3_output_path = "s3://output"
    parser_processor.parse(
        s3_input_path,
        s3_output_path,
    )
    expected_args = get_expected_args_all_parameters(
        parser_processor._current_job_name, s3_output_path, s3_input_path
    )
    sagemaker_session.process.assert_called_with(**expected_args)
    assert SEC_FILING_PARSER_JOB_NAME in parser_processor._current_job_name
    assert expected_args["app_specification"]["ImageUri"] == IMAGE_URI
    assert expected_args["inputs"][1]["S3Input"]["S3Uri"] == s3_input_path
    assert expected_args["output_config"]["Outputs"][0]["S3Output"]["S3Uri"] == s3_output_path


def get_expected_args_all_parameters(job_name, s3_output_path, s3_input_path=None):
    if s3_input_path:
        return {
            "inputs": [
                {
                    "InputName": "config",
                    "AppManaged": False,
                    "S3Input": {
                        "S3Uri": "mocked_s3_uri_from_upload_data",
                        "LocalPath": "/opt/ml/processing/input/config",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3CompressionType": "None",
                    },
                },
                {
                    "InputName": "data",
                    "AppManaged": False,
                    "S3Input": {
                        "S3Uri": s3_input_path,
                        "LocalPath": "/opt/ml/processing/input/data",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3CompressionType": "None",
                    },
                },
            ],
            "output_config": {
                "Outputs": [
                    {
                        "OutputName": "output-1",
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": s3_output_path,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "experiment_config": None,
            "job_name": job_name,
            "resources": {
                "ClusterConfig": {
                    "InstanceType": "ml.c5.xlarge",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30,
                }
            },
            "stopping_condition": None,
            "app_specification": {
                "ImageUri": "935494966801.dkr.ecr.us-west-2.amazonaws.com/{}:{}".format(
                    REPOSITORY, CONTAINER_IMAGE_VERSION
                )
            },
            "environment": None,
            "network_config": None,
            "role_arn": ROLE,
            "tags": None,
        }
    else:
        return {
            "inputs": [
                {
                    "InputName": "config",
                    "AppManaged": False,
                    "S3Input": {
                        "S3Uri": "mocked_s3_uri_from_upload_data",
                        "LocalPath": "/opt/ml/processing/input/config",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3CompressionType": "None",
                    },
                }
            ],
            "output_config": {
                "Outputs": [
                    {
                        "OutputName": "output-1",
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": s3_output_path,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "experiment_config": None,
            "job_name": job_name,
            "resources": {
                "ClusterConfig": {
                    "InstanceType": "ml.c5.xlarge",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30,
                }
            },
            "stopping_condition": None,
            "app_specification": {
                "ImageUri": "935494966801.dkr.ecr.us-west-2.amazonaws.com/{}:{}".format(
                    REPOSITORY, CONTAINER_IMAGE_VERSION
                )
            },
            "environment": None,
            "network_config": None,
            "role_arn": ROLE,
            "tags": None,
        }


def get_describe_response_inputs_and_outputs():
    return {
        "ProcessingInputs": get_expected_args_all_parameters(None, None, None)["inputs"],
        "ProcessingOutputConfig": get_expected_args_all_parameters(None, None, None)[
            "output_config"
        ],
    }
