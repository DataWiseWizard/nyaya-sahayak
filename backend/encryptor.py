"""
encryptor.py
------------
Handles encryption-at-rest for the ChromaDB persistent store and uploaded
PDFs.

Design:
  - Key is derived from a user passphrase via PBKDF2-HMAC-SHA256, never stored.
  - AES-256-CBC with a random IV per blob (IV prepended to ciphertext).
  - A random per-installation salt IS stored (salt is not secret; it just
    needs to be consistent across sessions) in `salt.bin` next to the vault.
  - ChromaDB's persistent store is a FOLDER (sqlite file + index files), not
    a single file — so the vault encrypts/decrypts a whole directory at
    once by zipping it in memory first, then AES-encrypting the zip bytes.
  - Provides a context manager `unlocked_vault()` that decrypts the DB to a
    temp folder for the duration of a session and securely wipes it
    afterwards (including on crashes, via atexit + signal handling).

This is NOT a substitute for full disk encryption — it protects the data
at rest (e.g. if the laptop is stolen while shut down), not a
running/unlocked session.
"""

from __future__ import annotations

import atexit
import gc
import io
import os
import secrets
import shutil
import signal
import sys
import json
import zipfile
from contextlib import contextmanager
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.padding import PKCS7

PBKDF2_ITERATIONS = 480_000  # OWASP 2024+ recommendation for SHA-256
KEY_LEN = 32  # AES-256
IV_LEN = 16
SALT_LEN = 16


class VaultEncryptor:
    def __init__(self, salt_path: Path):
        self.salt_path = Path(salt_path)

    # ---- key management -----------------------------------------------

    def _get_or_create_salt(self) -> bytes:
        if self.salt_path.exists():
            return self.salt_path.read_bytes()
        salt = secrets.token_bytes(SALT_LEN)
        self.salt_path.parent.mkdir(parents=True, exist_ok=True)
        self.salt_path.write_bytes(salt)
        return salt

    def derive_key(self, passphrase: str) -> bytes:
        salt = self._get_or_create_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_LEN,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    # ---- raw encrypt/decrypt -------------------------------------------

    def encrypt_bytes(self, key: bytes, plaintext: bytes) -> bytes:
        iv = secrets.token_bytes(IV_LEN)
        padder = PKCS7(algorithms.AES.block_size).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        return iv + ciphertext  # IV is not secret, safe to prepend

    def decrypt_bytes(self, key: bytes, blob: bytes) -> bytes:
        iv, ciphertext = blob[:IV_LEN], blob[IV_LEN:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded) + unpadder.finalize()

    # ---- single-file helpers (used for uploaded PDFs) ---------------------

    def encrypt_file(self, key: bytes, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self.encrypt_bytes(key, Path(src).read_bytes()))

    def decrypt_file(self, key: bytes, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self.decrypt_bytes(key, Path(src).read_bytes()))

    # ---- directory helpers (used for the ChromaDB store) ------------------

    def encrypt_dir(self, key: bytes, src_dir: Path, dst_file: Path) -> None:
        """Zips src_dir in memory, encrypts the zip bytes, writes to dst_file."""
        src_dir = Path(src_dir)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in src_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(src_dir))
        dst_file = Path(dst_file)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_bytes(self.encrypt_bytes(key, buf.getvalue()))

    def decrypt_dir(self, key: bytes, src_file: Path, dst_dir: Path) -> None:
        """Decrypts src_file and extracts the zip into dst_dir."""
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        plaintext = self.decrypt_bytes(key, Path(src_file).read_bytes())
        with zipfile.ZipFile(io.BytesIO(plaintext)) as zf:
            zf.extractall(dst_dir)

    @staticmethod
    def secure_delete(path: Path, retries: int = 5, retry_delay: float = 0.5) -> None:
        """Overwrite file contents before unlinking (best-effort on SSDs).

        On Windows, a just-closed ChromaDB/sqlite handle can take a moment
        to fully release — retries with a short delay before falling back
        to a plain (non-overwritten) delete rather than crashing the whole
        run. That fallback slightly weakens the "secure wipe" guarantee for
        that one file, but losing hours of ingestion work to a cleanup-step
        crash is worse.
        """
        import time

        path = Path(path)
        if not path.exists():
            return
        last_error = None
        for attempt in range(retries):
            try:
                length = path.stat().st_size
                with open(path, "r+b") as f:
                    f.write(secrets.token_bytes(length))
                    f.flush()
                    os.fsync(f.fileno())
                path.unlink()
                return
            except PermissionError as e:
                last_error = e
                gc.collect()
                time.sleep(retry_delay)
        # Still locked after retries — fall back to a plain delete attempt
        try:
            path.unlink()
            print(f"  [WARN] {path.name} was locked; deleted without secure "
                f"overwrite as a fallback.")
        except PermissionError:
            print(f"  [WARN] Could not delete {path} — it's still locked by "
                f"another process ({last_error}). You may need to delete "
                f"it manually once no Python process is using it.")

    @staticmethod
    def secure_delete_dir(dir_path: Path) -> None:
        """Securely wipes every file in a directory, then removes the tree."""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return
        for path in dir_path.rglob("*"):
            if path.is_file():
                VaultEncryptor.secure_delete(path)
        shutil.rmtree(dir_path, ignore_errors=True)


def open_vault(vault_encrypted_path: Path, salt_path: Path,
            passphrase: str, tmp_decrypted_dir: Path):
    """
    Decrypts the vault into `tmp_decrypted_dir` and registers crash-safety
    handlers (atexit + SIGINT/SIGTERM), but — unlike `unlocked_vault` —
    does NOT re-lock automatically when this function returns. Use this
    when you need the vault open across many operations in a long-lived
    process (e.g. a Streamlit app answering many questions in one session),
    where re-decrypting a multi-GB vault before every single question would
    be far too slow.

    Returns (tmp_decrypted_dir, close_fn). Call close_fn() explicitly when
    you're done (e.g. on an app "Lock Vault" button, or at shutdown) to
    re-encrypt and securely wipe the decrypted copy. If the process crashes
    or is killed first, the atexit/signal handlers do this automatically.
    """
    enc = VaultEncryptor(salt_path)
    key = enc.derive_key(passphrase)
    tmp_decrypted_dir = Path(tmp_decrypted_dir)

    if tmp_decrypted_dir.exists():
        shutil.rmtree(tmp_decrypted_dir, ignore_errors=True)

    if Path(vault_encrypted_path).exists():
        enc.decrypt_dir(key, vault_encrypted_path, tmp_decrypted_dir)
    else:
        tmp_decrypted_dir.mkdir(parents=True, exist_ok=True)

    state = {"closed": False}

    def _cleanup(*_args):
        if state["closed"]:
            return  # avoid double-cleanup if atexit AND explicit close both fire
        state["closed"] = True
        try:
            if tmp_decrypted_dir.exists():
                enc.encrypt_dir(key, tmp_decrypted_dir, vault_encrypted_path)
                VaultEncryptor.secure_delete_dir(tmp_decrypted_dir)
        finally:
            atexit.unregister(_cleanup)

    atexit.register(_cleanup)

    # signal.signal() only works when called from the interpreter's main
    # thread — fine for a plain CLI script, but Streamlit (and some other
    # frameworks) run the app in a worker thread, where this raises
    # ValueError. In that case I'll just skip SIGINT/SIGTERM handling and
    # rely on atexit alone for crash safety — atexit still fires on normal
    # interpreter shutdown either way.
    prev_sigint = prev_sigterm = None
    try:
        prev_sigint = signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(1)))
        prev_sigterm = signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(1)))
    except ValueError:
        pass  # not in main thread (e.g. running under Streamlit) — atexit still covers us

    def close_fn():
        _cleanup()
        if prev_sigint is not None:
            signal.signal(signal.SIGINT, prev_sigint)
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)

    return tmp_decrypted_dir, close_fn


@contextmanager
def unlocked_vault(vault_encrypted_path: Path, salt_path: Path,
                    passphrase: str, tmp_decrypted_dir: Path):
    """
    Convenience wrapper around open_vault/close_fn for short-lived,
    single-operation scripts (e.g. `rag_pipeline.py ingest`, one-off CLI
    queries) — decrypts for the life of the `with` block, then always
    re-locks when it exits, crash or not.

    For a long-lived interactive process that needs the vault open across
    many operations (e.g. the Streamlit app), use open_vault() directly
    instead so you're not re-decrypting a multi-GB store on every question.
    """
    tmp_decrypted_dir, close_fn = open_vault(
        vault_encrypted_path, salt_path, passphrase, tmp_decrypted_dir
    )
    try:
        yield tmp_decrypted_dir
    finally:
        close_fn()
        

def save_encrypted_chat(chat_list: list, passphrase: str, file_path: str = "data/chat_history.aes"):
    """Encrypts the chat history list into an AES-256 file."""
    if not chat_list:
        return
        
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000, backend=default_backend())
    key = kdf.derive(passphrase.encode())
    iv = os.urandom(16)
    
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(json.dumps(chat_list).encode()) + padder.finalize()
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    # We save the salt and IV at the top of the file so we can decrypt it later
    with open(file_path, "wb") as f:
        f.write(salt + iv + encrypted_data)

def load_encrypted_chat(passphrase: str, file_path: str = "data/chat_history.aes") -> list:
    """Decrypts the AES-256 file back into the chat history list."""
    if not Path(file_path).exists():
        return []
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    if len(data) < 32: 
        return []
        
    salt, iv, encrypted_data = data[:16], data[16:32], data[32:]
    
    try:
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000, backend=default_backend())
        key = kdf.derive(passphrase.encode())
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
        
        unpadder = padding.PKCS7(128).unpadder()
        decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()
        
        return json.loads(decrypted_data.decode())
    except Exception:
        # If the password is wrong or file is corrupted, return an empty history
        return []