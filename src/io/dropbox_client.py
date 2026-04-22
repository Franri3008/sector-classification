from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import settings
from src.logging_setup import get_logger
from src.utils.retry import retryable

if TYPE_CHECKING:
    import dropbox
logger = get_logger(__name__)


@dataclass
class DropboxFile:
    path: str
    size: int
    rev: str | None = None


class DropboxClient:
    def __init__(self, token: str | None = None) -> None:
        self._token = token or settings.dropbox_token.get_secret_value()
        self._dbx: dropbox.Dropbox | None = None

    def _client(self) -> dropbox.Dropbox:
        if self._dbx is None:
            if not self._token:
                raise RuntimeError(
                    "DROPBOX_TOKEN is empty — set it in .env before calling Dropbox."
                )
            import dropbox

            self._dbx = dropbox.Dropbox(self._token)
        return self._dbx

    @retryable()
    def download_to(self, dropbox_path: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("dropbox download %s -> %s", dropbox_path, local_path)
        md, resp = self._client().files_download(dropbox_path)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info("dropbox download done: %s bytes", md.size)
        return local_path

    @retryable()
    def list_folder_recursive(self, dropbox_path: str) -> list[DropboxFile]:
        import dropbox as _dbx

        client = self._client()
        result = client.files_list_folder(dropbox_path, recursive=True)
        files: list[DropboxFile] = []
        while True:
            for entry in result.entries:
                if isinstance(entry, _dbx.files.FileMetadata):
                    files.append(
                        DropboxFile(path=entry.path_display, size=entry.size, rev=entry.rev)
                    )
            if not result.has_more:
                break
            result = client.files_list_folder_continue(result.cursor)
        return files

    def sync_folder(
        self, dropbox_path: str, local_path: Path, *, force: bool = False
    ) -> list[Path]:
        local_path.mkdir(parents=True, exist_ok=True)
        remote_files = self.list_folder_recursive(dropbox_path)
        if not remote_files:
            logger.warning("dropbox folder is empty: %s", dropbox_path)
            return []
        base = dropbox_path.rstrip("/")
        base_cf = base.casefold()
        downloaded: list[Path] = []
        for f in remote_files:
            if not f.path.casefold().startswith(base_cf + "/"):
                logger.warning("skipping out-of-tree file: %s", f.path)
                continue
            rel = f.path[len(base) + 1 :]
            target = local_path / rel
            if target.exists() and (not force):
                if target.stat().st_size == f.size:
                    logger.debug("skip (already synced): %s", rel)
                    continue
                logger.info("size mismatch, re-downloading: %s", rel)
            self.download_to(f.path, target)
            downloaded.append(target)
        logger.info(
            "sync %s -> %s: %d downloaded, %d skipped",
            dropbox_path,
            local_path,
            len(downloaded),
            len(remote_files) - len(downloaded),
        )
        return downloaded
