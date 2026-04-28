"""GCS access for BIV-Priv and VISPR images.

All images are fetched directly from Cloud Storage — Flickr source URLs are
never used for retrieval (many are dead). This is an intentional leakage
demonstration: raw GCS object references are exposed, not a privacy-safe path.

Storage layout
--------------
BIV-Priv : gs://<bucket>/Biv_Priv/query_image_extracted/query_images/{n}.jpeg
VISPR     : gs://<bucket>/VISPR/vispr_image_extracted/val2017/{annotation_id}.jpg

The BIV-Priv record_id (e.g. bivpriv_04de2b39da6a) is derived from the numeric
filename as:  "bivpriv_" + md5("{n}.jpeg")[:12]
We reverse that by listing all BIV-Priv blobs once and building a hash→blob map.
"""

from __future__ import annotations

import hashlib
import datetime
import io

from src.config.settings import GCS_BUCKET_NAME

_BIVPRIV_PREFIX = "Biv_Priv/query_image_extracted/query_images/"
_VISPR_PREFIX = "VISPR/vispr_image_extracted/val2017/"
_project_name = 'project-a3ec92bf-1508-4ecb-9f0'

_storage_client = None
_bivpriv_hash_to_blob: dict[str, str] | None = None  # lazily populated


def _get_storage_client():
    global _storage_client
    if _storage_client is None:
        from google.cloud import storage
        _storage_client = storage.Client(_project_name)
    return _storage_client


def _get_bivpriv_hash_to_blob() -> dict[str, str]:
    """Build (once) a mapping: md5("{n}.jpeg")[:12] → full GCS blob name."""
    global _bivpriv_hash_to_blob
    if _bivpriv_hash_to_blob is not None:
        return _bivpriv_hash_to_blob

    client = _get_storage_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    _bivpriv_hash_to_blob = {}
    for blob in bucket.list_blobs(prefix=_BIVPRIV_PREFIX):
        filename = blob.name.split("/")[-1]
        if not filename:
            continue
        h = hashlib.md5(filename.encode()).hexdigest()[:12]
        _bivpriv_hash_to_blob[h] = blob.name
    return _bivpriv_hash_to_blob


def _bivpriv_record_to_blob(record_id: str) -> str | None:
    hash_part = record_id.removeprefix("bivpriv_")
    return _get_bivpriv_hash_to_blob().get(hash_part)


# ── Public URI helpers ────────────────────────────────────────────────────────

def get_gcs_uri(record_id: str) -> str:
    """Return the canonical gs:// URI for a BIV-Priv record_id."""
    if not GCS_BUCKET_NAME:
        return f"gs://<GCS_BUCKET_NAME>/{_BIVPRIV_PREFIX}<unknown>"
    blob_name = _bivpriv_record_to_blob(record_id)
    if blob_name is None:
        return f"gs://{GCS_BUCKET_NAME}/{_BIVPRIV_PREFIX}<unknown>"
    return f"gs://{GCS_BUCKET_NAME}/{blob_name}"


def get_vispr_gcs_uri(annotation_id: str) -> str:
    """Return the canonical gs:// URI for a VISPR annotation_id."""
    if not GCS_BUCKET_NAME:
        return f"gs://<GCS_BUCKET_NAME>/{_VISPR_PREFIX}{annotation_id}.jpg"
    return f"gs://{GCS_BUCKET_NAME}/{_VISPR_PREFIX}{annotation_id}.jpg"


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_blob(blob_name: str) -> bytes | None:
    client = _get_storage_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    buf = io.BytesIO()
    blob.download_to_file(buf)
    return buf.getvalue()


def download_image_bytes(record_id: str) -> bytes | None:
    """Download a BIV-Priv image from GCS by record_id. Returns None if not found."""
    blob_name = _bivpriv_record_to_blob(record_id)
    if blob_name is None:
        return None
    return _download_blob(blob_name)


def download_vispr_image_bytes(annotation_id: str) -> bytes | None:
    """Download a VISPR image from GCS by annotation_id. Returns None if not found."""
    if not annotation_id or annotation_id in ("None", ""):
        return None
    blob_name = f"{_VISPR_PREFIX}{annotation_id}.jpg"
    return _download_blob(blob_name)


def get_signed_url(record_id: str, expiry_minutes: int = 60) -> str:
    """Return a short-lived signed HTTPS URL for a BIV-Priv image."""
    blob_name = _bivpriv_record_to_blob(record_id)
    if blob_name is None:
        return ""
    client = _get_storage_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiry_minutes),
        method="GET",
    )
