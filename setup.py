# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

import os
from glob import glob

from setuptools import setup, find_packages


def read(fname):
    """
    Args:
        fname:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read("VERSION").strip()


# Declare minimal set for installation
required_packages = [
    "sagemaker>=2.48.0",
    "psutil>=5.8.0",
]


# Specific use case dependencies
extras = {
}
# Meta dependency groups
extras["all"] = [item for group in extras.values() for item in group]
# Tests specific dependencies (do not need to be included in 'all')
extras["test"] = (
    [
        extras["all"],
        "tox",
        "flake8",
        "pytest<6.1.0",
        "pytest-cov",
        "pytest-rerunfailures",
        "pytest-timeout",
        "pytest-xdist",
        "mock",
        "awslogs",
        "black",
        "wheel",
    ],
)

setup(
    name="smjsindustry",
    version=read_version(),
    description="Open source library for industry machine learning on Amazon SageMaker.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    long_description=read("README.rst"),
    author="Amazon Web Services",
    url="https://github.com/aws/sagemaker-jumpstart-industry-pack/",
    license="Apache License 2.0",
    keywords="ML Amazon AWS AI Jumpstart Industry",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=required_packages,
    extras_require=extras,
)