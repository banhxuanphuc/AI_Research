"""Lưu file gốc (PDF, TEI XML) vào MinIO / S3."""

import hashlib
import io

from minio import Minio

from research_scientist.core.config import get_settings


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ObjectStore:
    def __init__(self) -> None:
        s = get_settings()
        self.bucket = s.minio_bucket
        self._client = Minio(
            s.minio_endpoint,
            access_key=s.minio_root_user,
            secret_key=s.minio_root_password.get_secret_value(),
            secure=s.minio_secure,
        )

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)

    def bucket_ready(self) -> bool:
        return self._client.bucket_exists(self.bucket)

    def put(self, key: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            self.bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )
        return f"s3://{self.bucket}/{key}"

    def get(self, key: str) -> bytes:
        resp = self._client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()
