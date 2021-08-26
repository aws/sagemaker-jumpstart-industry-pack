# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Tests build_tabText module."""
from __future__ import absolute_import

import pandas as pd
from smjumpstart.finance.build_tabText import build_tabText


def test_build_tabText_by_quarter():
    tabular_df = pd.DataFrame(
        {
            "ticker": ["ticker1", "ticker2"],
            "date": ["2019-01-01", "2020-01-01"],
            "price": [2000.00, 100.00],
        }
    )

    text_df = pd.DataFrame(
        {
            "ticker": ["ticker1", "ticker2"],
            "date": ["2019-02-01", "2020-02-02"],
            "doc": ["doc1", "doc2"],
        }
    )

    joined = build_tabText(
        tabular_df,
        "ticker",
        "date",
        text_df,
        "ticker",
        "date",
    )
    assert set(joined.columns) == set(["ticker", "date_x", "date_y", "doc", "price"])
    assert joined.loc[0, "doc"] == text_df.loc[0, "doc"]
    assert joined.loc[1, "doc"] == text_df.loc[1, "doc"]


def test_build_tabText_by_year():
    tabular_df = pd.DataFrame(
        {
            "ticker1": ["ticker1", "ticker2"],
            "date1": ["2019-01-01", "2020-01-01"],
            "price": [2000.00, 100.00],
        }
    )

    text_df = pd.DataFrame(
        {
            "ticker2": ["ticker1", "ticker2"],
            "date2": ["2019-02-01", "2020-02-02"],
            "doc": ["doc1", "doc2"],
        }
    )

    joined = build_tabText(
        tabular_df,
        "ticker1",
        "date1",
        text_df,
        "ticker2",
        "date2",
        freq="Y",
    )
    assert set(joined.columns) == set(["ticker1", "date1", "ticker2", "date2", "doc", "price"])
    assert joined.loc[0, "doc"] == text_df.loc[0, "doc"]
    assert joined.loc[1, "doc"] == text_df.loc[1, "doc"]
