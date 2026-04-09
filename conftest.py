# Root conftest.py — prevents pytest from importing the ComfyUI plugin
# __init__.py (which uses relative imports only valid inside ComfyUI).
# This file deliberately stays empty of fixtures.
