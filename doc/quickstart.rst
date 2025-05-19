#######################
Quickstart Guide
#######################

This guide will help you get started with the Amazon SageMaker Python SDK quickly.

.. contents::
   :local:
   :depth: 2

*******************
Prerequisites
*******************

Before you begin:

1. **Install the SDK**:

   .. code:: bash

      pip install sagemaker

2. **Configure AWS credentials**:

   .. code:: bash

      aws configure

3. **Create an S3 bucket** for storing training data and model artifacts:

   .. code:: python

      import sagemaker
      
      sagemaker_session = sagemaker.Session()
      bucket = sagemaker_session.default_bucket()

*******************
Train Your First Model
*******************

Let's train a simple classification model using the built-in XGBoost algorithm:

.. code:: python

   import sagemaker
   from sagemaker.xgboost import XGBoost
   import pandas as pd
   import numpy as np
   
   # Initialize a SageMaker session
   session = sagemaker.Session()
   
   # Prepare sample data
   data = pd.DataFrame({
       'feature1': np.random.randn(100),
       'feature2': np.random.randn(100),
       'label': [0, 1] * 50
   })
   
   # Split data into train and validation sets
   train_data = data.sample(frac=0.8)
   validation_data = data.drop(train_data.index)
   
   # Save data to CSV files
   train_data.to_csv('train.csv', index=False, header=False)
   validation_data.to_csv('validation.csv', index=False, header=False)
   
   # Upload data to S3
   prefix = 'xgboost-demo'
   train_s3 = session.upload_data('train.csv', key_prefix=prefix + '/train')
   validation_s3 = session.upload_data('validation.csv', key_prefix=prefix + '/validation')
   
   # Define the XGBoost estimator
   xgb = XGBoost(
       entry_point='script.py',  # Optional custom script
       framework_version='1.5-1',
       hyperparameters={
           'max_depth': 5,
           'eta': 0.2,
           'objective': 'binary:logistic',
           'num_round': 100
       },
       instance_count=1,
       instance_type='ml.m5.xlarge',
       role=sagemaker.get_execution_role()
   )
   
   # Train the model
   xgb.fit({
       'train': train_s3,
       'validation': validation_s3
   })

*******************
Deploy and Get Predictions
*******************

After training, deploy your model to a SageMaker endpoint:

.. code:: python

   # Deploy the model to an endpoint
   predictor = xgb.deploy(
       initial_instance_count=1,
       instance_type='ml.m5.xlarge'
   )
   
   # Prepare test data
   test_data = pd.DataFrame({
       'feature1': np.random.randn(10),
       'feature2': np.random.randn(10)
   })
   
   # Make predictions
   predictions = predictor.predict(test_data.values)
   print(predictions)
   
   # Clean up when done
   predictor.delete_endpoint()

*******************
Using Your Own Algorithm
*******************

You can also bring your own training script:

.. code:: python

   from sagemaker.pytorch import PyTorch
   
   # Define a PyTorch estimator with your custom script
   pytorch_estimator = PyTorch(
       entry_point='train.py',  # Your PyTorch training script
       source_dir='code',       # Directory containing your code
       framework_version='1.12.0',
       py_version='py38',
       instance_count=1,
       instance_type='ml.p3.2xlarge',
       role=sagemaker.get_execution_role(),
       hyperparameters={
           'epochs': 10,
           'batch-size': 64,
           'learning-rate': 0.001
       }
   )
   
   # Train the model
   pytorch_estimator.fit({'training': 's3://bucket/training-data'})

*******************
Local Mode Development
*******************

Test your training and inference code locally before deploying to the cloud:

.. code:: python

   # Train locally
   local_estimator = PyTorch(
       entry_point='train.py',
       source_dir='code',
       framework_version='1.12.0',
       py_version='py38',
       instance_count=1,
       instance_type='local',  # Use 'local_gpu' if you have a GPU
       role=sagemaker.get_execution_role()
   )
   
   # Train using local data
   local_estimator.fit({'training': 'file:///path/to/local/data'})
   
   # Deploy locally
   local_predictor = local_estimator.deploy(
       initial_instance_count=1,
       instance_type='local'
   )
   
   # Make predictions
   local_predictions = local_predictor.predict(test_data)
   
   # Clean up
   local_predictor.delete_endpoint()

*******************
Next Steps
*******************

Now that you've completed the quickstart guide, you can explore more advanced features:

- Use built-in algorithms for specific tasks
- Train with popular deep learning frameworks
- Set up hyperparameter tuning jobs
- Create ML pipelines for automation
- Monitor and debug your models
- Deploy to production with advanced options

Check out the :doc:`overview` for more details on these topics.