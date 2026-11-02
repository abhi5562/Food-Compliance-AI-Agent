import os
from uuid import uuid4

import boto3
from dotenv import load_dotenv


load_dotenv()


def upload_image(image_bytes: bytes, content_type: str, prefix: str = "inspections"):
    """Upload one evidence image and return its permanent S3 reference."""
    extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    if not image_bytes:
        raise ValueError("The image is empty.")
    if content_type not in extensions:
        raise ValueError("Use a JPEG, PNG, or WebP image.")

    bucket = os.environ["S3_EVIDENCE_BUCKET"]
    key = f"{prefix}/{uuid4().hex}{extensions[content_type]}"
    session = boto3.Session(profile_name=os.getenv("AWS_PROFILE"))
    s3 = session.client("s3", region_name=os.getenv("AWS_REGION"))
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=image_bytes,
            ContentType=content_type,
        )
    finally:
        s3.close()

    return {"bucket": bucket, "key": key}


def read_image(bucket: str, key: str) -> bytes:
    """Read an evidence image using its stored S3 reference."""
    session = boto3.Session(profile_name=os.getenv("AWS_PROFILE"))
    s3 = session.client("s3", region_name=os.getenv("AWS_REGION"))
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        with response["Body"] as body:
            return body.read()
    finally:
        s3.close()
