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

import pytest
from mock import Mock, patch

from sagemaker.model import Model
from sagemaker.model_life_cycle import ModelLifeCycle
from sagemaker.session import Session


@pytest.fixture()
def sagemaker_session():
    session = Mock(spec=Session)
    session.sagemaker_client.describe_model_package_group.return_value = {
        "ModelPackageGroupStatus": "Completed"
    }
    session.sagemaker_client.create_model_package.return_value = {"ModelPackageArn": "arn:aws:sagemaker:model-package/test"}
    return session


def test_model_register_with_model_life_cycle(sagemaker_session):
    """Test that model.register() correctly handles ModelLifeCycle objects"""
    model = Model(
        image_uri="test-image",
        role="test-role",
        sagemaker_session=sagemaker_session
    )
    
    # Create a ModelLifeCycle object
    model_life_cycle = ModelLifeCycle(
        stage="Development",
        stage_status="In-Progress",
        stage_description="Development In Progress"
    )
    
    # Expected dictionary representation
    expected_life_cycle_dict = {
        "Stage": "Development",
        "StageStatus": "In-Progress",
        "StageDescription": "Development In Progress"
    }
    
    # Call register with the ModelLifeCycle object
    model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="test-group",
        model_life_cycle=model_life_cycle
    )
    
    # Get the args that were passed to create_model_package
    create_model_package_args = sagemaker_session.sagemaker_client.create_model_package.call_args[1]
    
    # Verify that ModelLifeCycle was properly converted to a dictionary
    assert "ModelLifeCycle" in create_model_package_args
    assert create_model_package_args["ModelLifeCycle"] == expected_life_cycle_dict