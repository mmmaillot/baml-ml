"""Thin name adapter for the historical synchronous test API.

No numerical calculations or error handling are implemented here. Generated
BAML models and functions are used directly. A missing export fails loudly.
"""
import baml_sdk


class SyncClient:
    def __getattr__(self, name):
        return getattr(baml_sdk, name + "_sync")


b = SyncClient()
types = baml_sdk
