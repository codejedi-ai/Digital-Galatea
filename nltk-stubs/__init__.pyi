"""Type stubs for nltk"""
from typing import Any

class data:
    @staticmethod
    def find(resource_name: str, paths: list[str] = ...]) -> str: ...

def download(info_or_id: str, download_dir: str = ..., quiet: bool = ..., force: bool = ...,
             raise_on_error: bool = ..., halt_on_error: bool = ...) -> bool: ...

def word_tokenize(text: str) -> list[str]: ...

