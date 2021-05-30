import os
import subprocess as sp

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# Return the git revision as a string
# taken from numpy/numpy
def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, env=env).communicate()[0]
    return out

def git_version():
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def git_count():
    try:
        out = _minimal_ext_cmd(["git", "rev-list", "--count", "HEAD"])
        GIT_COUNT = out.strip().decode("ascii")
    except OSError:
        GIT_COUNT = "Unknown"

    return GIT_COUNT

def _get_git_cmd(fcn):
    cwd = os.getcwd()

    # go to the main directory
    fdir = os.path.dirname(os.path.abspath(__file__))
    maindir = os.path.abspath(os.path.join(fdir, ".."))
    # maindir = fdir # os.path.join(fdir, "..")
    os.chdir(maindir)

    # get git version
    res = fcn()

    # restore the cwd
    os.chdir(cwd)
    return res

def get_version():
    fdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(fdir, "_version.txt")
    if os.path.exists(fname):
        with open(fname, "r") as f:
            version = f.read().strip()
            return version

    if ISRELEASED:
        version = VERSION

    # unreleased version
    else:
        ngit_short = 7  # how many letters should be included from the git version
        GIT_REVISION = _get_git_cmd(git_version)
        GIT_REVISION_SHORT = GIT_REVISION[:ngit_short]
        GIT_COUNT = _get_git_cmd(git_count)
        num_int_git_short = len(str(int("f" * ngit_short, 16)))
        git_rev_format = f"%0{num_int_git_short}d"
        version = VERSION + ".dev" + GIT_COUNT + (git_rev_format % int(GIT_REVISION_SHORT, 16))

    with open(fname, "w") as f:
        f.write(version)
    return version

if __name__ == "__main__":
    print(get_version())
