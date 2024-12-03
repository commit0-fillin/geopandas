"""Git implementation of _version.py."""
import errno
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import functools

def get_keywords() -> Dict[str, str]:
    """Get the keywords needed to look up the version information."""
    # These strings will be replaced by git during git-archive.
    # They may be used by the package to determine the version.
    git_refnames = "$Format:%d$"
    git_full = "$Format:%H$"
    git_date = "$Format:%ci$"
    keywords = {"refnames": git_refnames, "full": git_full, "date": git_date}
    return keywords

class VersioneerConfig:
    """Container for Versioneer configuration parameters."""
    VCS: str
    style: str
    tag_prefix: str
    parentdir_prefix: str
    versionfile_source: str
    verbose: bool

def get_config() -> VersioneerConfig:
    """Create, populate and return the VersioneerConfig() object."""
    cfg = VersioneerConfig()
    cfg.VCS = "git"
    cfg.style = "pep440"
    cfg.tag_prefix = "v"
    cfg.parentdir_prefix = "geopandas-"
    cfg.versionfile_source = "geopandas/_version.py"
    cfg.verbose = False
    return cfg

class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""
LONG_VERSION_PY: Dict[str, str] = {}
HANDLERS: Dict[str, Dict[str, Callable]] = {}

def register_vcs_handler(vcs: str, method: str) -> Callable:
    """Create decorator to mark a method as the handler of a VCS."""
    def decorate(f: Callable) -> Callable:
        if vcs not in HANDLERS:
            HANDLERS[vcs] = {}
        HANDLERS[vcs][method] = f
        return f
    return decorate

def run_command(commands: List[str], args: List[str], cwd: Optional[str]=None, verbose: bool=False, hide_stderr: bool=False, env: Optional[Dict[str, str]]=None) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    p = None
    for c in commands:
        try:
            dispcmd = str([c] + args)
            # remember shell=False, so use git.cmd on windows, not just git
            p = subprocess.Popen([c] + args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=(subprocess.PIPE if hide_stderr else None))
            break
        except EnvironmentError:
            e = sys.exc_info()[1]
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % dispcmd)
                print(e)
            return None, None
    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return None, None
    stdout = p.communicate()[0].strip().decode()
    if p.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % dispcmd)
            print("stdout was %s" % stdout)
        return None, p.returncode
    return stdout, p.returncode

def versions_from_parentdir(parentdir_prefix: str, root: str, verbose: bool) -> Dict[str, Any]:
    """Try to determine the version from the parent directory name.

    Source tarballs conventionally unpack into a directory that includes both
    the project name and a version string. We will also support searching up
    two directory levels for an appropriately named parent directory
    """
    rootdirs = []

    for i in range(3):
        dirname = os.path.basename(root)
        if dirname.startswith(parentdir_prefix):
            return {"version": dirname[len(parentdir_prefix):],
                    "full-revisionid": None,
                    "dirty": False, "error": None, "date": None}
        else:
            rootdirs.append(root)
            root = os.path.dirname(root)

    if verbose:
        print("Tried directories %s but none started with prefix %s" %
              (str(rootdirs), parentdir_prefix))
    raise NotThisMethod("rootdir doesn't start with parentdir_prefix")

@register_vcs_handler('git', 'get_keywords')
def git_get_keywords(versionfile_abs: str) -> Dict[str, str]:
    """Extract version information from the given file."""
    # the code embedded in _version.py can just fetch the value of these
    # keywords. When used from setup.py, we don't want to import _version.py,
    # so we do it with a regexp instead. This function is not used from
    # _version.py.
    keywords = {}
    try:
        f = open(versionfile_abs, "r")
        for line in f.readlines():
            if line.strip().startswith("git_refnames ="):
                mo = re.search(r'=\s*"(.*)"', line)
                if mo:
                    keywords["refnames"] = mo.group(1)
            if line.strip().startswith("git_full ="):
                mo = re.search(r'=\s*"(.*)"', line)
                if mo:
                    keywords["full"] = mo.group(1)
            if line.strip().startswith("git_date ="):
                mo = re.search(r'=\s*"(.*)"', line)
                if mo:
                    keywords["date"] = mo.group(1)
        f.close()
    except EnvironmentError:
        pass
    return keywords

@register_vcs_handler('git', 'keywords')
def git_versions_from_keywords(keywords: Dict[str, str], tag_prefix: str, verbose: bool) -> Dict[str, Any]:
    """Get version information from git keywords."""
    if not keywords:
        raise NotThisMethod("no keywords at all, weird")
    date = keywords.get("date")
    if date is not None:
        # git-2.2.0 added "%cI", which expands to an ISO-8601 format
        date = date.strip().replace(" ", "T", 1).replace(" ", "", 1)
    refnames = keywords["refnames"].strip()
    if refnames.startswith("$Format"):
        if verbose:
            print("keywords are unexpanded, not using")
        raise NotThisMethod("unexpanded keywords, not a git-archive tarball")
    refs = set([r.strip() for r in refnames.strip("()").split(",")])
    # starting in git-1.8.3, tags are listed as "tag: foo-1.0" instead of
    # just "foo-1.0". If we see a "tag: " prefix, prefer those.
    TAG = "tag: "
    tags = set([r[len(TAG):] for r in refs if r.startswith(TAG)])
    if not tags:
        # Either we're using git < 1.8.3, or there really are no tags. We use
        # a heuristic: assume all version tags have a digit. The old git %d
        # expansion behaves like git log --decorate=short and strips out the
        # refs/heads/ and refs/tags/ prefixes that would let us distinguish
        # between branches and tags. By ignoring refnames without digits, we
        # filter out many common branch names like "release" and
        # "stabilization", as well as "HEAD" and "master".
        tags = set([r for r in refs if re.search(r'\d', r)])
        if verbose:
            print("discarding '%s', no digits" % ",".join(refs - tags))
    if verbose:
        print("likely tags: %s" % ",".join(sorted(tags)))
    for ref in sorted(tags):
        # sorting will prefer e.g. "2.0" over "2.0rc1"
        if ref.startswith(tag_prefix):
            r = ref[len(tag_prefix):]
            if verbose:
                print("picking %s" % r)
            return {"version": r,
                    "full-revisionid": keywords["full"].strip(),
                    "dirty": False, "error": None,
                    "date": date}
    # no suitable tags, so version is "0+unknown", but full hex is still there
    if verbose:
        print("no suitable tags, using unknown + full revision id")
    return {"version": "0+unknown",
            "full-revisionid": keywords["full"].strip(),
            "dirty": False, "error": "no suitable tags", "date": None}

@register_vcs_handler('git', 'pieces_from_vcs')
def git_pieces_from_vcs(tag_prefix: str, root: str, verbose: bool, runner: Callable=run_command) -> Dict[str, Any]:
    """Get version from 'git describe' in the root of the source tree.

    This only gets called if the git-archive 'subst' keywords were *not*
    expanded, and _version.py hasn't already been rewritten with a short
    version string, meaning we're inside a checked out source tree.
    """
    pass

def plus_or_dot(pieces: Dict[str, Any]) -> str:
    """Return a + if we don't already have one, else return a ."""
    pass

def render_pep440(pieces: Dict[str, Any]) -> str:
    """Build up version string, with post-release "local version identifier".

    Our goal: TAG[+DISTANCE.gHEX[.dirty]] . Note that if you
    get a tagged build and then dirty it, you'll get TAG+0.gHEX.dirty

    Exceptions:
    1: no tags. git_describe was just HEX. 0+untagged.DISTANCE.gHEX[.dirty]
    """
    pass

def render_pep440_branch(pieces: Dict[str, Any]) -> str:
    """TAG[[.dev0]+DISTANCE.gHEX[.dirty]] .

    The ".dev0" means not master branch. Note that .dev0 sorts backwards
    (a feature branch will appear "older" than the master branch).

    Exceptions:
    1: no tags. 0[.dev0]+untagged.DISTANCE.gHEX[.dirty]
    """
    pass

def pep440_split_post(ver: str) -> Tuple[str, Optional[int]]:
    """Split pep440 version string at the post-release segment.

    Returns the release segments before the post-release and the
    post-release version number (or -1 if no post-release segment is present).
    """
    pass

def render_pep440_pre(pieces: Dict[str, Any]) -> str:
    """TAG[.postN.devDISTANCE] -- No -dirty.

    Exceptions:
    1: no tags. 0.post0.devDISTANCE
    """
    pass

def render_pep440_post(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]+gHEX] .

    The ".dev0" means dirty. Note that .dev0 sorts backwards
    (a dirty tree will appear "older" than the corresponding clean one),
    but you shouldn't be releasing software with -dirty anyways.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]
    """
    pass

def render_pep440_post_branch(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]+gHEX[.dirty]] .

    The ".dev0" means not master branch.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]+gHEX[.dirty]
    """
    pass

def render_pep440_old(pieces: Dict[str, Any]) -> str:
    """TAG[.postDISTANCE[.dev0]] .

    The ".dev0" means dirty.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]
    """
    pass

def render_git_describe(pieces: Dict[str, Any]) -> str:
    """TAG[-DISTANCE-gHEX][-dirty].

    Like 'git describe --tags --dirty --always'.

    Exceptions:
    1: no tags. HEX[-dirty]  (note: no 'g' prefix)
    """
    pass

def render_git_describe_long(pieces: Dict[str, Any]) -> str:
    """TAG-DISTANCE-gHEX[-dirty].

    Like 'git describe --tags --dirty --always -long'.
    The distance/hash is unconditional.

    Exceptions:
    1: no tags. HEX[-dirty]  (note: no 'g' prefix)
    """
    pass

def render(pieces: Dict[str, Any], style: str) -> Dict[str, Any]:
    """Render the given version pieces into the requested style."""
    if pieces["error"]:
        return {"version": "unknown",
                "full-revisionid": pieces.get("full-revisionid", ""),
                "dirty": None,
                "error": pieces["error"],
                "date": None}

    if not style or style == "default":
        style = "pep440"  # default

    if style == "pep440":
        rendered = render_pep440(pieces)
    elif style == "pep440-pre":
        rendered = render_pep440_pre(pieces)
    elif style == "pep440-post":
        rendered = render_pep440_post(pieces)
    elif style == "pep440-old":
        rendered = render_pep440_old(pieces)
    elif style == "git-describe":
        rendered = render_git_describe(pieces)
    elif style == "git-describe-long":
        rendered = render_git_describe_long(pieces)
    else:
        raise ValueError("unknown style '%s'" % style)

    return {"version": rendered, "full-revisionid": pieces["full-revisionid"],
            "dirty": pieces["dirty"], "error": None,
            "date": pieces.get("date")}

def get_versions() -> Dict[str, Any]:
    """Get version information or return default if unable to do so."""
    # I am in _version.py, which lives at ROOT/VERSIONFILE_SOURCE. If we have
    # __file__, we can work backwards from there to the root. Some
    # py2exe/bbfreeze/non-CPython implementations don't do __file__, in which
    # case we can only use expanded keywords.

    cfg = get_config()
    verbose = cfg.verbose

    try:
        return git_versions_from_keywords(get_keywords(), cfg.tag_prefix, verbose)
    except NotThisMethod:
        pass

    try:
        root = os.path.realpath(__file__)
        # versionfile_source is the relative path from the top of the source
        # tree (where the .git directory might live) to this file. Invert
        # this to find the root from __file__.
        for i in cfg.versionfile_source.split('/'):
            root = os.path.dirname(root)
    except NameError:
        return {"version": "0+unknown", "full-revisionid": None,
                "dirty": None,
                "error": "unable to find root of source tree",
                "date": None}

    try:
        pieces = git_pieces_from_vcs(cfg.tag_prefix, root, verbose)
        return render(pieces, cfg.style)
    except NotThisMethod:
        pass

    try:
        if cfg.parentdir_prefix:
            return versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
    except NotThisMethod:
        pass

    return {"version": "0+unknown", "full-revisionid": None,
            "dirty": None,
            "error": "unable to compute version", "date": None}
