import json

import boto3

from config import (
    storage_bucket,
    storage_endpoint,
    storage_access_key,
    storage_secret_key,
    storage_region,
)


class ObjectStorage:

    def __init__(self):
        required = {
            "BUCKET": storage_bucket,
            "ENDPOINT": storage_endpoint,
            "ACCESS_KEY_ID": storage_access_key,
            "SECRET_ACCESS_KEY": storage_secret_key,
            "REGION": storage_region,
        }

        missing = [
            name
            for name, value in required.items()
            if not value
        ]

        if missing:
            raise RuntimeError(
                "Object storage is not configured. "
                f"Missing: {', '.join(missing)}"
            )

        self.bucket = storage_bucket

        self.client = boto3.client(
            "s3",
            endpoint_url=storage_endpoint,
            aws_access_key_id=storage_access_key,
            aws_secret_access_key=storage_secret_key,
            region_name=storage_region,
        )

    def upload_file(self, file_path, object_key):
        self.client.upload_file(
            file_path,
            self.bucket,
            object_key,
        )

    def download_file(self, object_key, file_path):
        self.client.download_file(
            self.bucket,
            object_key,
            file_path,
        )

    def delete_file(self, object_key):
        self.client.delete_object(
            Bucket=self.bucket,
            Key=object_key,
        )

    def put_json(self, object_key, data):
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=json.dumps(data).encode("utf-8"),
            ContentType="application/json",
        )

    def get_json(self, object_key):
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=object_key,
        )

        return json.loads(
            response["Body"].read().decode("utf-8")
        )

    def list_objects(self, prefix):
        keys = []
        continuation_token = None

        while True:
            params = {
                "Bucket": self.bucket,
                "Prefix": prefix,
            }

            if continuation_token:
                params["ContinuationToken"] = (
                    continuation_token
                )

            response = self.client.list_objects_v2(
                **params
            )

            keys.extend(
                item["Key"]
                for item in response.get(
                    "Contents",
                    []
                )
            )

            if not response.get(
                "IsTruncated",
                False
            ):
                break

            continuation_token = response.get(
                "NextContinuationToken"
            )

            if not continuation_token:
                break

        return keys