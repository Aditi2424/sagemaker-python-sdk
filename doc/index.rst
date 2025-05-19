###########################
Amazon SageMaker Python SDK
###########################

.. image:: https://readthedocs.org/projects/sagemaker/badge/?version=stable
   :target: https://sagemaker.readthedocs.io/en/stable/
   :alt: Documentation Status

Amazon SageMaker Python SDK is an open source library for training and deploying machine learning models on Amazon SageMaker.

.. raw:: html

   <div class="sagemaker-overview">
     <div class="overview-card">
       <h3>Train</h3>
       <p>Train ML models with built-in algorithms or your own code</p>
     </div>
     <div class="overview-card">
       <h3>Deploy</h3>
       <p>Deploy models for real-time or batch inference</p>
     </div>
     <div class="overview-card">
       <h3>MLOps</h3>
       <p>Manage the complete ML lifecycle</p>
     </div>
   </div>

.. contents::
   :local:
   :depth: 2

************
Installation
************

Install the latest stable version:

.. code:: bash

   pip install sagemaker

For more details and options, see the :doc:`installation guide <installation>`.

***********
Quick Start
***********

Train and deploy your first model in minutes:

.. code:: python

   import sagemaker
   from sagemaker.sklearn import SKLearn
   
   # Initialize a SageMaker session
   sagemaker_session = sagemaker.Session()
   
   # Define an estimator
   estimator = SKLearn(
       entry_point='script.py',
       framework_version='1.0-1',
       instance_type='ml.m5.xlarge',
       instance_count=1,
       role=sagemaker.get_execution_role()
   )
   
   # Train the model
   estimator.fit({'training': 's3://your-bucket/path/to/training/data'})
   
   # Deploy the model
   predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')
   
   # Make predictions
   predictions = predictor.predict(data)
   
   # Clean up when done
   predictor.delete_endpoint()

For a more comprehensive guide, see the :doc:`quickstart guide <quickstart>`.

*****************
SageMaker Core
*****************

The SageMaker Python SDK provides several high-level abstractions for working with Amazon SageMaker:

- **ModelTrainer**: New interface encapsulating training on SageMaker
- **Estimators**: Encapsulate training on SageMaker
- **Models**: Encapsulate built ML models
- **Predictors**: Provide real-time inference against a SageMaker endpoint
- **Session**: Provides methods for working with SageMaker resources
- **Transformers**: Encapsulate batch transform jobs for inference
- **Processors**: Encapsulate processing jobs for data processing

.. toctree::
    :maxdepth: 1

    overview
    v2
    api/index

*****************
Key Components
*****************

Training
========

Train machine learning models at scale with built-in algorithms or your own custom code.

**SageMaker Debugger**

Automatically detect anomalies during training, monitor resource utilization, and gain insights into your model's behavior.

.. toctree::
   :maxdepth: 1

   amazon_sagemaker_debugger

**Frameworks Support**

Train and deploy models using popular deep learning frameworks:

- TensorFlow
- PyTorch
- MXNet
- Scikit-learn
- XGBoost
- Hugging Face
- And more...

.. toctree::
    :maxdepth: 1

    frameworks/index

**Built-in Algorithms**

Use Amazon SageMaker's optimized implementations of common ML algorithms:

.. toctree::
    :maxdepth: 1

    algorithms/index

Inference
=========

Deploy your models for real-time or batch inference:

- Real-time endpoints for low-latency predictions
- Batch transform for large datasets
- Multi-model endpoints for hosting multiple models
- Serverless inference for cost-effective deployment

MLOps
=====

Manage the complete machine learning lifecycle:

**Feature Store**

Store, share, and manage features and their metadata:

.. toctree::
   :maxdepth: 1

   amazon_sagemaker_featurestore

**Model Monitoring**

Automatically detect concept drift and data quality issues:

.. toctree::
    :maxdepth: 1

    amazon_sagemaker_model_monitoring

**Processing**

Perform data processing tasks at scale:

.. toctree::
    :maxdepth: 1

    amazon_sagemaker_processing

**Model Building Pipeline**

Orchestrate your ML workflows:

.. toctree::
    :maxdepth: 1

    amazon_sagemaker_model_building_pipeline

**Experiments**

Track and compare training runs:

.. toctree::
    :maxdepth: 1

    experiments/index

**Workflows**

Integrate with Airflow and Kubernetes:

.. toctree::
    :maxdepth: 1

    workflows/index

JumpStart
=========

Access pre-trained models and solution templates:

- Deploy foundation models with minimal code
- Fine-tune models on your data
- Use solution templates for common ML use cases

.. toctree::
    :maxdepth: 1

    doc_utils/pretrainedmodels

*****************
SDK Features
*****************

Intelligent Defaults
===================

The SageMaker Python SDK provides intelligent defaults that simplify the machine learning workflow:

- Automatic resource selection
- Pre-configured environments for popular frameworks
- Sensible parameter defaults based on best practices
- Simplified API for common tasks

Local Mode
=========

Develop and test your training and inference code locally before deploying to the cloud:

.. code:: python

   # Run training locally
   estimator = PyTorch(
       entry_point='train.py',
       instance_type='local',
       ...
   )
   
   # Deploy locally
   predictor = estimator.deploy(
       initial_instance_count=1,
       instance_type='local',
       ...
   )

Remote Functions
===============

Execute Python functions remotely on SageMaker infrastructure:

.. code:: python

   from sagemaker.remote_function import remote
   
   @remote
   def train_model(hyperparameters):
       # This code runs on SageMaker
       import numpy as np
       # Your training code here
       return model_artifacts
   
   # Call the function - executes on SageMaker
   model = train_model({"learning_rate": 0.01})

.. toctree::
    :maxdepth: 1

    remote_function/sagemaker.remote_function

*****************
Additional Resources
*****************

- `GitHub Repository <https://github.com/aws/sagemaker-python-sdk>`_
- `Amazon SageMaker Developer Guide <https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html>`_
- `API Reference <https://sagemaker.readthedocs.io/en/stable/api/index.html>`_
- `AWS Machine Learning Blog <https://aws.amazon.com/blogs/machine-learning/>`_
- `Example Notebooks <https://github.com/aws/amazon-sagemaker-examples>`_