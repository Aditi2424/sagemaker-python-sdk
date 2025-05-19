#######################
Installation Guide
#######################

This guide provides detailed instructions for installing the Amazon SageMaker Python SDK.

************
Requirements
************

- Python 3.6 or later
- pip 20.0 or later

*******************
Standard Installation
*******************

Install the latest stable version:

.. code:: bash

   pip install sagemaker

To upgrade an existing installation:

.. code:: bash

   pip install --upgrade sagemaker

*******************
Optional Dependencies
*******************

The SageMaker Python SDK has several optional dependencies that provide additional functionality:

.. code:: bash

   # For using SciPy-related functionality
   pip install sagemaker[scipy]
   
   # For local mode with TensorFlow
   pip install sagemaker[local,tensorflow]
   
   # For local mode with PyTorch
   pip install sagemaker[local,pytorch]
   
   # For local mode with MXNet
   pip install sagemaker[local,mxnet]

*******************
Development Installation
*******************

To install the package from source for development:

.. code:: bash

   git clone https://github.com/aws/sagemaker-python-sdk.git
   cd sagemaker-python-sdk
   pip install -e ".[develop]"

*******************
AWS Configuration
*******************

To use the SageMaker Python SDK, you need to set up your AWS credentials:

1. **Using AWS CLI**:

   .. code:: bash

      aws configure

2. **Using Environment Variables**:

   .. code:: bash

      export AWS_ACCESS_KEY_ID=your-access-key
      export AWS_SECRET_ACCESS_KEY=your-secret-key
      export AWS_DEFAULT_REGION=your-region

3. **Using a Credentials File**:

   Create or edit ``~/.aws/credentials``:

   .. code::

      [default]
      aws_access_key_id = your-access-key
      aws_secret_access_key = your-secret-key

   Create or edit ``~/.aws/config``:

   .. code::

      [default]
      region = your-region

*******************
Verify Installation
*******************

To verify that the SageMaker Python SDK is installed correctly:

.. code:: python

   import sagemaker
   print(sagemaker.__version__)

*******************
Troubleshooting
*******************

**Dependency Conflicts**

If you encounter dependency conflicts, try installing in a virtual environment:

.. code:: bash

   python -m venv sagemaker-env
   source sagemaker-env/bin/activate  # On Windows: sagemaker-env\Scripts\activate
   pip install sagemaker

**Permission Issues**

Ensure your AWS IAM role has the necessary permissions for SageMaker:

- ``AmazonSageMakerFullAccess`` for full access
- Custom policies with specific permissions for more restricted access

**Region Availability**

Verify that SageMaker is available in your selected AWS region:
https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/