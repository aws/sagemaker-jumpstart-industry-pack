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
"""Tests utils module."""
from __future__ import absolute_import

import pytest
from smjsindustry.finance.utils import get_freq_label


@pytest.mark.parametrize(
    "date_value", ["2020-05-01", "2020-05", "2020", "2020/05/01", "2020|6", 2020]
)
@pytest.mark.parametrize("freq", ["Y", "Q", "M", "W", "D", "T", "y"])
def test_get_freq_label(date_value, freq):
    if date_value == 2020:
        with pytest.raises(Exception) as error:
            get_freq_label(date_value, freq)
            assert "The date column needs to be string" in str(error.value)
    elif freq == "T":
        with pytest.raises(ValueError, match=r"^frequency .* not supported$"):
            get_freq_label(date_value, freq)
    elif date_value == "2020/05/01" or date_value == "2020|6":
        with pytest.raises(ValueError, match=r"^Date needs to be in .* format when freq is .$"):
            get_freq_label(date_value, freq)
    elif freq == "Y" or freq == "y":
        actual = get_freq_label(date_value, freq)
        expected = "2020"
        assert actual == expected
    elif freq == "Q" and date_value != "2020":
        actual = get_freq_label(date_value, freq)
        expected = "2020Q2"
        assert actual == expected
    elif freq == "M" and date_value != "2020":
        actual = get_freq_label(date_value, freq)
        expected = "2020M5"
        assert actual == expected
    elif freq == "W" and date_value not in ("2020", "2020-05"):
        actual = get_freq_label(date_value, freq)
        expected = "2020W18"
        assert actual == expected
    elif freq == "D" and date_value == "2020-05-01":
        actual = get_freq_label(date_value, freq)
        expected = "2020-05-01"
        assert actual == expected
    else:
        with pytest.raises(ValueError, match=r"^Date needs to be in .* format when freq is .$"):
            get_freq_label(date_value, freq)
