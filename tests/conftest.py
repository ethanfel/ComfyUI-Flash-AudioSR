# conftest.py — isolate tests from the ComfyUI plugin package.
# The root __init__.py uses relative imports that only work when ComfyUI
# loads the plugin as a package. We add the plugin root to sys.path here
# so that model_loader (and other top-level modules) can be imported
# directly without triggering the relative-import __init__.py.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
