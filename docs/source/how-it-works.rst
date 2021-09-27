How It Works
============

#. Use the SageMaker JumpStart Industry Python SDK's API operations
   in a Studio notebook. The SDK uploads the API configuration and input datasets
   (if necessary) to an Amazon S3 bucket that users can encrypt using Amazon VPC.
#. The API spins up a container to run the API operations as a SageMaker processing job.
#. SageMaker copies the input dataset and the API config to the containers.
#. SageMaker calls the container entrypoint command and runs the processing job.
#. After the processing job has completed, SageMaker copies the result from
   the processing containers to the Amazon S3 bucket.
   SageMaker terminates the cluster.
#. User downloads the result from the Amazon S3 bucket to the Studio notebook kernel
   and continues the Studio journey.
