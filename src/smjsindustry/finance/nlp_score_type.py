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
"""The NLP score type module of SageMaker JumpStart Industry."""
from __future__ import absolute_import
from typing import List

NLPSCORE_NO_WORD_LIST = ["sentiment", "polarity", "readability"]


class NLPScoreType:
    """Initializes an ``NLPScoreType`` instance.

    It wraps score names and their corresponding word lists used for NLP scoring.

    It provides an organized standard for passing required data to an NLPScorerConfig
    and defines several constants, such as ``POSITIVE`` and ``READABILITY``, which can be used
    to perform NLP scoring using SageMaker JumpStart Industry's internal word lists.

    A single ``NLPScoreType`` or a list of ``NLPScoreTypes`` is required
    when initializing an :class:`~smjsindustry.NLPScorerConfig`. Passing the data required by
    the :class:`~smjsindustry.NLPScorerConfig` via ``NLPScoreTypes`` ensures that any potential
    errors which could affect the creation of the config are caught
    at the earliest possible stage.

    To create an ``NLPScoreType`` using SageMaker JumpStart Industry's internal word lists, use
    an ``NLPScoreType`` constant (such as ``NLPScoreType.POSITIVE``) for the ``score_name``
    argument, and either ``[]`` or ``None`` for the ``word_list`` argument.

    Args:
        score_name (str):
            A name that describes the overall topic represented by the words in
            the word_list argument. For example, if the ``word_list`` argument is
            ``["promising", "prodigy", "talented", "adept"]``,
            the ``score_name`` argument could be ``"talent"``.

            SageMaker JumpStart Industry has internal
            word lists corresponding to the following
            ``score_name`` values:
            ``NLPScoreType.POSITIVE``, ``NLPScoreType.NEGATIVE``, ``NLPScoreType.POLARITY``,
            ``NLPScoreType.CERTAINTY``, ``NLPScoreType.UNCERTAINTY``, ``NLPScoreType.FRAUD``,
            ``NLPScoreType.LITIGIOUS``, ``NLPScoreType.RISK``, ``NLPScoreType.SAFE``,
            ``NLPScoreType.READABILITY``, ``NLPScoreType.SENTIMENT``.
        word_list (List[str]):
            A list of words corresponding to the topic indicated by ``score_name``.

            The following ``score_names`` values require the ``word_list`` argument to be ``None``
            (the remaining score names require ``word_list`` to be ``[]``):
            ``NLPScoreType.POLARITY``, ``NLPScoreType.READABILITY``, ``NLPScoreType.SENTIMENT``.

    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    CERTAINTY = "certainty"
    UNCERTAINTY = "uncertainty"
    RISK = "risk"
    SAFE = "safe"
    LITIGIOUS = "litigious"
    FRAUD = "fraud"
    SENTIMENT = "sentiment"
    POLARITY = "polarity"
    READABILITY = "readability"

    DEFAULT_SCORE_TYPES = [
        POSITIVE,
        NEGATIVE,
        CERTAINTY,
        UNCERTAINTY,
        RISK,
        SAFE,
        LITIGIOUS,
        FRAUD,
        SENTIMENT,
        POLARITY,
        READABILITY,
    ]

    def __init__(self, score_name: str, word_list: List[str]):
        """Initializes an ``NLPScoreType`` instance."""
        score_name = score_name.lower()
        self._score_name = score_name
        if score_name in NLPSCORE_NO_WORD_LIST:
            if word_list is not None:
                raise TypeError(
                    "NLPScoreType with score_name {} requires its word_list "
                    "argument to be None.".format(score_name)
                )
        else:
            if not isinstance(word_list, list):
                raise TypeError(
                    "NLPScoreType with score_name {} requires its word_list "
                    "argument to be a list.".format(score_name)
                )
            if score_name in NLPScoreType.DEFAULT_SCORE_TYPES:
                if word_list and any(not isinstance(word, str) for word in word_list):
                    raise TypeError("word_list argument must contain only strings.")
            else:
                if not word_list:
                    raise ValueError(
                        "NLPScoreType with custom score_name {} requires its word_list "
                        "argument to be a non-empty list.".format(score_name)
                    )
                if any(not isinstance(word, str) for word in word_list):
                    raise TypeError("word_list argument must contain only strings.")
        self._word_list = word_list

    @property
    def score_name(self) -> str:
        """Gets the string of the ``score_name`` argument."""
        return self._score_name

    @property
    def word_list(self) -> List[str]:
        """Gets the string of the ``word_list`` argument."""
        return self._word_list
