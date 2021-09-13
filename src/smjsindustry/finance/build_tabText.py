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
"""The module that builds a TabText dataframe."""
from __future__ import absolute_import

import pandas as pd
from smjsindustry.finance.utils import get_freq_label


JUMPSTART_NORMALIZED_DATE = "jumpstart-normalized-date"


def build_tabText(
    tabular_df: pd.DataFrame,
    tabular_key: str,
    tabular_date_column: str,
    text_df: pd.DataFrame,
    text_key: str,
    text_date_column: str,
    how: str = "inner",
    freq: str = "Q",
) -> pd.DataFrame:
    """Builds a TabText dataframe by joining the columns in the tabular and text dataframes.

    It joins a tabular dataframe and a text dataframe to create a TabText data.
    Each row of the two dataframes must be uniquely defined by a composite key
    consisting of a key and a date column. After the date columns are normalized
    according to the given freq, the two dataframes can be merged using
    the key column and the normalized date column.

    Args:
        tabular_df (pandas.DataFrame): The tabular dataframe to be joined, requiring a date column.
        tabular_key (str): The tabular dataframe's key column to be joined on.
        tabular_date_column (str): The tabular dataframe's date column to be joined on,
            in a format of ``"yyyy-mm-dd"``, ``"yyyy-mm"``, or ``"yyyy"``.
        text_df (pandas.DataFrame): The text dataframe to be joined, requiring a date column.
        text_key (str): The text dataframe's key column to be joined on.
        text_date_column (str): The text dataframe's date column to be joined on,
            in a format of ``"yyyy-mm-dd"``, ``"yyyy-mm"``, or ``"yyyy"``.
        how (str): The type of join to be performed, possible values:
            ``{'left', 'right', 'outer', 'inner'}`` (default: ``'inner'``).
        freq (str): specify how the date field should be joined,
            by year, quarter, month, week or day. Possible values:
            ``{'Y', 'Q', 'M', 'W', 'D'}`` (default: ``'Q'``).

    Returns:
        pandas.DataFrame: The joined Dataframe object.
    """
    if tabular_date_column and text_date_column:
        tabular_df[JUMPSTART_NORMALIZED_DATE] = tabular_df[tabular_date_column]
        for i in range(len(tabular_df)):
            date_value = tabular_df.loc[i, tabular_date_column]
            freq_label = get_freq_label(date_value, freq)
            tabular_df.loc[i, JUMPSTART_NORMALIZED_DATE] = freq_label
        text_df[JUMPSTART_NORMALIZED_DATE] = text_df[text_date_column]
        for i in range(len(text_df)):
            date_value = text_df.loc[i, text_date_column]
            freq_label = get_freq_label(date_value, freq)
            text_df.loc[i, JUMPSTART_NORMALIZED_DATE] = freq_label
        joined = pd.merge(
            tabular_df,
            text_df,
            left_on=[tabular_key, JUMPSTART_NORMALIZED_DATE],
            right_on=[text_key, JUMPSTART_NORMALIZED_DATE],
            how=how,
        )
        tabular_df.drop(columns=[JUMPSTART_NORMALIZED_DATE], inplace=True)
        text_df.drop(columns=[JUMPSTART_NORMALIZED_DATE], inplace=True)
        joined.drop(columns=[JUMPSTART_NORMALIZED_DATE], inplace=True)
    else:
        joined = pd.merge(tabular_df, text_df, left_on=tabular_key, right_on=text_key, how=how)
    return joined
