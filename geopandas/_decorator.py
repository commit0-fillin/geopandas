from textwrap import dedent
from typing import Callable, Union

def doc(*docstrings: Union[str, Callable], **params) -> Callable:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.
    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : str or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """
    def decorator(func):
        # Store original docstrings
        func._docstring_components = list(docstrings)

        # Concatenate and process docstrings
        doc = func.__doc__ or ""
        for docstring in docstrings:
            if callable(docstring):
                doc += dedent(docstring.__doc__ or "")
            else:
                doc += dedent(docstring)

        # Perform string substitution
        func.__doc__ = doc.format(**params)

        return func
    return decorator
