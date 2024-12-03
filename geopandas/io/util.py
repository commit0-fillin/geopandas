"""Vendored, cut down version of pyogrio/util.py for use with fiona"""
import re
import sys
from urllib.parse import urlparse

def vsi_path(path: str) -> str:
    """
    Ensure path is a local path or a GDAL-compatible vsi path.

    """
    parsed = _parse_uri(path)
    return _construct_vsi_path(*parsed)
SCHEMES = {'file': 'file', 'zip': 'zip', 'tar': 'tar', 'gzip': 'gzip', 'http': 'curl', 'https': 'curl', 'ftp': 'curl', 's3': 's3', 'gs': 'gs', 'az': 'az', 'adls': 'adls', 'adl': 'adls', 'hdfs': 'hdfs', 'webhdfs': 'webhdfs'}
CURLSCHEMES = {k for k, v in SCHEMES.items() if v == 'curl'}

def _parse_uri(path: str):
    """
    Parse a URI

    Returns a tuples of (path, archive, scheme)

    path : str
        Parsed path. Includes the hostname and query string in the case
        of a URI.
    archive : str
        Parsed archive path.
    scheme : str
        URI scheme such as "https" or "zip+s3".
    """
    parsed = urlparse(path)
    scheme = parsed.scheme.lower()
    
    if '+' in scheme:
        archive_scheme, inner_scheme = scheme.split('+', 1)
        if archive_scheme in SCHEMES and inner_scheme in SCHEMES:
            return parsed.path, f"/vsi{SCHEMES[archive_scheme]}/{parsed.netloc}{parsed.path}", f"{archive_scheme}+{inner_scheme}"
    
    if scheme in SCHEMES:
        if scheme in CURLSCHEMES:
            return f"{parsed.netloc}{parsed.path}{parsed.query}", None, scheme
        return parsed.path, None, scheme
    
    return path, None, 'file'

def _construct_vsi_path(path, archive, scheme) -> str:
    """Convert a parsed path to a GDAL VSI path"""
    if archive:
        return f"/vsi{SCHEMES[scheme.split('+')[0]]}/{archive}"
    elif scheme in CURLSCHEMES:
        return f"/vsi{SCHEMES[scheme]}/{scheme}://{path}"
    elif scheme in SCHEMES:
        return f"/vsi{SCHEMES[scheme]}/{path}"
    else:
        return path
