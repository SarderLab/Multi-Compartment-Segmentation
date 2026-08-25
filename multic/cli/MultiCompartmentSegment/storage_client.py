"""First-party storage API client, replacing girder_client for this container's I/O.

Talks to STORAGE_API_URL (retire-girder-dependency Task Group 3.2 — not built as an HTTP surface
yet, so nothing here has been run against a real server) using JOB_AUTH_TOKEN (Task Group 5, a real
per-job-scoped JWT) as Bearer auth. The exact endpoint paths below are a guess at the contract, not
a confirmed API — update once Task Group 3.2 exists for real.
"""
import os
import re

import requests

_CONTENT_DISPOSITION_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^;"\n]+)"?')


def _filename_from_response(resp, fallback):
    """Reads the real filename off Content-Disposition rather than the caller guessing an
    extension (e.g. assuming .svs, which is wrong for .tif/.ndpi/etc. slides)."""
    header = resp.headers.get('Content-Disposition', '')
    match = _CONTENT_DISPOSITION_FILENAME_RE.search(header)
    return match.group(1).strip() if match else fallback


class StorageClient:
    def __init__(self, base_url, token):
        self.base_url = base_url.rstrip('/')
        self.token = token

    def _headers(self, **extra):
        return {'Authorization': f'Bearer {self.token}', **extra}

    def _download(self, path, dest_path):
        resp = requests.get(f'{self.base_url}{path}', headers=self._headers(), stream=True)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return resp

    def download_input(self, item_id, dest_dir):
        """Downloads this item's WSI into dest_dir using its real filename (from the server's
        Content-Disposition header), not a guessed extension. Returns the local path."""
        os.makedirs(dest_dir, exist_ok=True)
        tmp_path = os.path.join(dest_dir, f'.{item_id}.download')
        resp = self._download(f'/items/{item_id}/file', tmp_path)
        filename = _filename_from_response(resp, fallback=f'{item_id}.bin')
        dest_path = os.path.join(dest_dir, filename)
        os.replace(tmp_path, dest_path)
        return dest_path

    def download_model(self, model_id, dest_path):
        self._download(f'/models/{model_id}', dest_path)

    def post(self, path, parameters=None, data=None):
        """girder_client.GirderClient-compatible signature — existing `gc.post(path=..., parameters=..., data=...)`
        call sites in this pipeline's own code (e.g. IterativePredict_1X.py) work unchanged against this client."""
        resp = requests.post(
            f'{self.base_url}/{path}', params=parameters, data=data,
            headers=self._headers(**{'Content-Type': 'application/json'}))
        resp.raise_for_status()
        return resp.json()

    def upload_result_file(self, item_id, filename, local_path):
        with open(local_path, 'rb') as f:
            resp = requests.post(f'{self.base_url}/items/{item_id}/files/{filename}', data=f, headers=self._headers())
        resp.raise_for_status()
        return resp.json()


def upload_result_files(storage_client: StorageClient, output_files, item_id):
    """Replaces upload_assetstore_files.uploadFilesToOriginalFolder — that function wrote directly to
    Girder's assetstore filesystem path (and carried a hardcoded fallback admin API key, see
    upload_assetstore_files.py:64). This is a plain per-file upload against the storage API instead."""
    import os
    for path in output_files:
        storage_client.upload_result_file(item_id, os.path.basename(path), path)
