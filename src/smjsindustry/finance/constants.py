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
"""This module defines constants used in SageMaker JumpStart Industry."""
from __future__ import print_function, absolute_import

JACCARD_SUMMARIZER = "JACCARD_SUMMARIZER"
KMEDOIDS_SUMMARIZER = "KMEDOIDS_SUMMARIZER"
NLP_SCORER = "NLP_SCORER"
LOAD_DATA = "LOAD_DATA"
SEC_XML_FILING_PARSER = "SEC_XML_FILING_PARSER"
SUMMARIZER_JOB_NAME = "jumpstart-gecko-summarize"
NLP_SCORE_JOB_NAME = "jumpstart-gecko-nlp-score"
SEC_FILING_RETRIEVAL_JOB_NAME = "jumpstart-gecko-sec-retrieve"
SEC_FILING_PARSER_JOB_NAME = "jumpstart-gecko-sec-parse"
SUPPORTED_SEC_FORMS = [
    "10-K",
    "10-Q",
    "497",
    "497K",
    "8-K",
    "S-3ASR",
    "N-1A",
    "485BXT",
    "485BPOS",
    "485APOS",
    "S-3",
    "S-3/A",
    "DEF 14A",
    "SC 13D",
    "SC 13D/A",
]
KMEDOIDS_SUMMARIZER_METRIC_VALUES = ["euclidean", "cosine", "dot-product"]
KMEDOIDS_SUMMARIZER_INIT_VALUES = ["random", "heuristic", "k-medoids++", "build"]
IMAGE_CONFIG_FILE = "image_uri.json"
REPOSITORY = "jumpstart-gecko"
ECR_URI_TEMPLATE = "{account_id}.dkr.ecr.{region}.amazonaws.com/{repository}"
CONTAINER_IMAGE_VERSION = "1.0.0"
