import logging
from aiohttp import ClientError
import boto3
import os


def upload_to_s3(bucket_name: str, file_name: str, object_name: str = None):
    """
    Upload a file to an S3 bucket

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_name (str): File to upload.
        object_name (str, optional): S3 object name. If not specified then file_name is used.
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def main():
    bucket_name = "bmi-predictor-models"

    directory_name = "checkpoints"

    # List all files in the directory
    files = os.listdir(directory_name)

    # Upload each file
    for file_name in files:
        file_path = os.path.join(directory_name, file_name)
        upload_to_s3(bucket_name, file_path)


if __name__ == "__main__":
    main()
