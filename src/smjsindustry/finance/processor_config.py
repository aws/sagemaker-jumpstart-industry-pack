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
"""The smjsindustry Finance processing job config module.

These classes assist in providing the necessary information to
configure smjsindustry Finance Processors.
"""
from __future__ import print_function, absolute_import

from abc import ABC, abstractmethod
import re
import logging
from typing import Any, Dict, List, Set, Union
from smjsindustry.finance.constants import (
    JACCARD_SUMMARIZER,
    KMEDOIDS_SUMMARIZER,
    NLP_SCORER,
    LOAD_DATA,
    SUPPORTED_SEC_FORMS,
)
from smjsindustry.finance.nlp_score_type import NLPScoreType

logger = logging.getLogger()


class FinanceProcessorConfig(ABC):
    """Config class for SageMaker Finance processors."""

    def __init__(self, processor_type: str):
        """Initializes a configuration for SageMaker Finance processor.

        Args:
            processor_type (str): An unique dataset key.
        """
        self._processor_type = processor_type

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Returns the config to be passed to a SageMaker Finance Processor instance."""
        return None

    @property
    def processor_type(self) -> str:
        """Gets processor_type."""
        return self._processor_type


class JaccardSummarizerConfig(FinanceProcessorConfig):
    """Config class for JaccardSummarizer.

    It specifies parameters required by the Jaccard summarization algorithm.

    The aim of the Jaccard summarizer is to extract the main thematic sentences
    out of the document. This is an Extractive summarization.
    Abstractive summarization is eschewed because SEC filings and
    legal financial text have strict meanings; thus, small changes
    in sentence structure may alter the legal meaning of the text.

    Extractive summarization also works for very long documents
    that cannot be easily processed with abstractive summarization.

    """

    def __init__(
        self,
        summary_size: int = 0,
        summary_percentage: float = 0,
        max_tokens: int = 0,
        cutoff: float = 0,
        vocabulary: Set[str] = None,
    ):
        """Initializes a ``JaccardSummarizerConfig`` instance.

        Args:
            summary_size (int): The max number of sentences in the summary (default: 0).
            summary_percentage (float): The number of sentences in the summary
                should not exceed a :code:`summary_percentage` of the sentences
                in the original text (default: 0).
            max_tokens (int): The max number of tokens in the summary (default: 0).
            cutoff (float): The similarity cut off (default: 0).
            vocabulary (Set[str]): A set of sentiment words (default: None).

        """
        super().__init__(JACCARD_SUMMARIZER)
        size_arguments = [summary_size, summary_percentage, max_tokens, cutoff]
        size_argument_count = sum([1 if arg else 0 for arg in size_arguments])
        if size_argument_count != 1:
            raise ValueError(
                "Only one summary size related argument can be specified, "
                "choose to specify one from summary_size, summary_percentage, max_tokens, cutoff."
            )
        self._summary_size = summary_size
        self._summary_percentage = summary_percentage
        self._max_tokens = max_tokens
        self._cutoff = cutoff
        self._vocabulary = vocabulary

    def get_config(self) -> Dict[str, Union[str, int, float, Set[str]]]:
        """Returns the config to be passed to a SageMaker Finance Summarizer instance."""
        return {
            "processor_type": self.processor_type,
            "summary_size": self.summary_size,
            "summary_percentage": self.summary_percentage,
            "max_tokens": self.max_tokens,
            "cutoff": self.cutoff,
            "vocabulary": self.vocabulary,
        }

    @property
    def summary_size(self) -> int:
        """Gets summary_size."""
        return self._summary_size

    @property
    def summary_percentage(self) -> float:
        """Gets summary_percentage."""
        return self._summary_percentage

    @property
    def max_tokens(self) -> int:
        """Gets max_tokens."""
        return self._max_tokens

    @property
    def cutoff(self) -> float:
        """Gets cutoff."""
        return self._cutoff

    @property
    def vocabulary(self) -> Set[str]:
        """Gets vocabulary."""
        return self._vocabulary


class KMedoidsSummarizerConfig(FinanceProcessorConfig):
    """Config class for KMedoidsSummarizer.

    It specifies parameters required by the k-medoids summarization algorithm.

    The k-medoids summarizer is an extractive summarizer and
    uses a k-medoids based approach. First, it creates sentence embeddings
    using gensim’s Doc2Vec. Second, k-medoids clustering is performed on
    the sentence vectors.

    Note that this summarizer uses k-medoids instead of
    k-means clustering. Whereas k-means minimizes the total squared error
    from a central position in each cluster (centroid), k-medoids minimizes
    the sum of dissimilarities between vectors in a cluster. One of
    the vectors of each cluster is designated as the representative vector,
    which is called the medoids.

    The collection of medoids
    and the m sentences in the document closest to the cluster medoids
    is returned as the summary. The goal of this summarizer is different
    from the Jaccard Summarizer. The KMedoidsSummarizer aims to pick up
    peripheral sentences, not just the main theme of the document, in case
    there are items of importance that are buried in sentences different
    from the main theme. This is Extractive summarization.
    Abstractive summarization is eschewed because SEC filings and
    legal financial text have strict meanings and small changes
    in sentence structure may alter the legal meaning of the text.
    Extractive summarization also works for very long documents that
    cannot be easily processed with abstractive summarization.

    """

    def __init__(
        self,
        summary_size: int,
        vector_size: int = 100,
        min_count: int = 0,
        epochs: int = 60,
        metric: str = "euclidean",
        init: str = "heuristic",
    ):
        """Initializes a ``KMedoidsSummarizerConfig`` instance.

        Args:
            summary_size (int): The number of sentences to be extracted.
            vector_size (int): The embedding dimensions (default: 100).
            min_count (int): The minimal word occurrences to be included (default: 0).
            epochs (int): The number of epochs in a training (default: 60).
            metric (str): The distance metric to use.
                Possible values are 'euclidean', 'cosine', 'dot-product' (default: "euclidean").
            init (str): The value specifies medoid initialization method.
                Possible values are 'random', 'heuristic', 'k-medoids++', 'build'
                (default: 'heuristic').
        """
        super().__init__(KMEDOIDS_SUMMARIZER)
        self._summary_size = summary_size
        self._vector_size = vector_size
        self._min_count = min_count
        self._epochs = epochs
        self._metric = metric
        self._init = init

    def get_config(self) -> Dict[str, Union[str, int]]:
        """Returns the config to be passed to a SageMaker Finance Summarizer instance."""
        return {
            "processor_type": self.processor_type,
            "summary_size": self.summary_size,
            "vector_size": self.vector_size,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "metric": self.metric,
            "init": self.init,
        }

    @property
    def summary_size(self) -> int:
        """Gets summary_size."""
        return self._summary_size

    @property
    def vector_size(self) -> int:
        """Gets vector_size."""
        return self._vector_size

    @property
    def min_count(self) -> int:
        """Gets min_count."""
        return self._min_count

    @property
    def epochs(self) -> int:
        """Gets epochs."""
        return self._epochs

    @property
    def metric(self) -> str:
        """Gets metric."""
        return self._metric

    @property
    def init(self) -> str:
        """Gets init."""
        return self._init


class NLPScorerConfig(FinanceProcessorConfig):
    """Config class for NLPScorer.

    It specifies the word lists and their corresponding names that
    will be used when performing NLP scoring on a document.
    """

    def __init__(self, nlp_score_types: List[NLPScoreType]):
        """Initializes a ``NLPScorerConfig`` instance.

        Args:
            nlp_score_types (List[NLPScoreType]):
                The score types that will be used for NLP scoring.
        """
        super().__init__(NLP_SCORER)
        self._config = {}
        self._config["processor_type"] = self.processor_type
        self._config["score_types"] = {}
        if not isinstance(nlp_score_types, list):
            nlp_score_types = [nlp_score_types]
        for score_type in nlp_score_types:
            if not isinstance(score_type, NLPScoreType):
                raise TypeError(
                    "An NLPScorerConfig must be initialized with "
                    "either a single NLPScoreType object, or "
                    "a list of NLPScoreType objects."
                )
            self._config["score_types"][score_type.score_name] = score_type.word_list

    def get_config(self) -> Dict[str, Union[str, Dict[str, List[str]]]]:
        """Returns the config to be passed to a SageMaker Finance NLPScorer instance."""
        return self._config


class EDGARDataSetConfig(FinanceProcessorConfig):
    """Config class for loading SEC filings from SEC EDGAR.

    It specifies the details of SEC filings required by the DataLoader.
    """

    def __init__(
        self,
        tickers_or_ciks: List[str] = None,
        form_types: List[str] = None,
        filing_date_start: str = None,
        filing_date_end: str = None,
        email_as_user_agent: str = None,
    ):
        """Initializes a ``EDGARDataSetConfig`` instance.

        Args:

            tickers_or_ciks (List[str]): A list of stock tickers or CIKs. | e.g. ['amzn']
            form_types (List[str]): A list of SEC form types. The supported form types are
                10-K, 10-Q, 8-K, 497, 497K, S-3ASR, N-1A, 485BXT, 485BPOS, 485APOS, S-3,
                S-3/A, DEF 14A, SC 13D and SC 13D/A. | e.g. ['10-K']
            filing_date_start (str): The starting filing date in the format of
                'YYYY-MM-DD'. | e.g. '2021-01-01'
            filing_date_end (str): The ending filing date in the format of
                'YYYY-MM-DD'. | e.g. '2021-12-31'
            email_as_user_agent (str): The user email used as a user_agent for SEC EDGAR
                HTTP requests. | e.g. "gecko_demo_user@amazon.com"

        Raises:
            TypeError:
                if tickers_or_ciks (List[str]) is not a list OR any item in the list is not a string
                if form_types (List[str]) is not a list OR any item in the list is not a string
                if filing_date_start (str) is not a string
                if filing_date_end (str) is not a string
                if email_as_user_agent (str) is not a string
            ValueError:
                if any item in the form_types (List[str]) is not from SUPPORTED_SEC_FORMS
                if filing_date_start (str) is not in the format of 'YYYY-MM-DD'
                if filing_date_end (str) is not in the format of 'YYYY-MM-DD'
                if email_as_user_agent (str) is not a valid email address

        """
        super().__init__(LOAD_DATA)
        if (
            not tickers_or_ciks
            or not isinstance(tickers_or_ciks, list)
            or any(not isinstance(ticker_or_cik, str) for ticker_or_cik in tickers_or_ciks)
        ):
            raise TypeError("EDGARDataSetConfig requires tickers_or_ciks to be a list of strings.")
        if (
            not form_types
            or not isinstance(form_types, list)
            or any(not isinstance(form_type, str) for form_type in form_types)
        ):
            raise TypeError("EDGARDataSetConfig requires form_types to be a list of strings.")
        for form_type in form_types:
            if form_type.upper() not in SUPPORTED_SEC_FORMS:
                raise ValueError(f"{form_type} not supported.")
        if not isinstance(filing_date_start, str):
            raise TypeError("EDGARDataSetConfig requires filing_date_start to be a string.")
        if not bool(re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", filing_date_start)):
            raise ValueError(
                "EDGARDataSetConfig requires filing_date_start in the format of 'YYYY-MM-DD'."
            )
        if not isinstance(filing_date_end, str):
            raise TypeError("EDGARDataSetConfig requires filing_date_end to be a string.")
        if not bool(re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", filing_date_end)):
            raise ValueError(
                "EDGARDataSetConfig requires filing_date_end in the format of 'YYYY-MM-DD'."
            )
        if not isinstance(email_as_user_agent, str):
            raise TypeError("EDGARDataSetConfig requires email_as_user_agent to be a string.")
        if not re.match(r"^[A-Za-z0-9\.\+_-]+@[A-Za-z0-9\._-]+\.[a-zA-Z]*$", email_as_user_agent):
            raise ValueError(
                "EDGARDataSetConfig requires email_as_user_agent to be a valid email address."
            )
        self._tickers_or_ciks = tickers_or_ciks
        self._form_types = form_types
        self._filing_date_start = filing_date_start
        self._filing_date_end = filing_date_end
        self._email_as_user_agent = email_as_user_agent
        logger.info(
            "Use of Gecko is subject to the SEC terms and conditions "
            "governing the EDGAR database. You should conduct your own "
            "review of the terms to make sure they are acceptable for your "
            "use case before proceeding."
        )

    def get_config(self):
        """Returns config to be passed to a SageMaker Finance DataLoader instance."""
        return {
            "processor_type": self.processor_type,
            "tickers_or_ciks": self.tickers_or_ciks,
            "form_types": self.form_types,
            "filing_date_start": self.filing_date_start,
            "filing_date_end": self.filing_date_end,
            "email_as_user_agent": self.email_as_user_agent,
        }

    @property
    def tickers_or_ciks(self):
        """Gets tickers_or_ciks."""
        return self._tickers_or_ciks

    @property
    def form_types(self):
        """Gets form_types."""
        return self._form_types

    @property
    def filing_date_start(self):
        """Gets filing_date_start."""
        return self._filing_date_start

    @property
    def filing_date_end(self):
        """Gets filing_date_end."""
        return self._filing_date_end

    @property
    def email_as_user_agent(self):
        """Gets email_as_user_agent."""
        return self._email_as_user_agent
