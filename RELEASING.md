# Releasing

Publishing a release to [PyPI](https://pypi.org/project/shirokumas/) means creating a
GitHub Release. The `Upload Python Package` workflow takes it from there.

This document is for maintainers. It needs push access to the repository. There is no
upload credential to hold: PyPI trusts the workflow itself, through a trusted publisher
registered against this repository, `python-publish.yml`, and the `pypi` environment.
The runner mints a short-lived token per run, so nothing long-lived exists to leak.

## Where the version comes from

The version is derived from the Git tag by
[setuptools-scm](https://setuptools-scm.readthedocs.io/), and from nothing else. There
is no version to edit in `pyproject.toml`, and `shirokumas.__version__` reads back the
metadata the build produced rather than being the thing that feeds it.

| state of the checkout | reported version |
| --- | --- |
| exactly on `v0.1.0`, clean tree | `0.1.0` |
| a few commits past it | `0.1.1.dev3+g1a2b3c4` |
| on a tagged commit with a dirty tree | `0.1.1.dev0+g1a2b3c4.d20260830` |

The `v` prefix is stripped by setuptools-scm's default tag pattern, so keep naming tags
`vX.Y.Z`.

## Steps

### 1. Get the release content onto `main`

Whatever `main` points at when the tag is created is what gets built, so merge
everything the release should contain first — including any change to the
`Development Status` classifier in `pyproject.toml`.

```console
$ git switch main
$ git pull
```

Check that the `Testing Python Package` workflow is green for that commit.

### 2. Create the tag

The existing tags are annotated, so keep it that way:

```console
$ git tag -a v0.1.0 -m "Release 0.1.0"
```

### 3. Confirm the version before pushing

This is the step that catches mistakes while they are still free to fix:

```console
$ git describe --tags
v0.1.0
$ uvx --from setuptools-scm python -m setuptools_scm
0.1.0
```

The second command must print the release version exactly. If it prints a `.devN`
version instead, the working tree is dirty — setuptools-scm rolls a dirty tree forward
to a development version of the *next* release. Commit or stash, then check again.

```console
$ git push origin v0.1.0
```

### 4. Publish the GitHub Release

```console
$ gh release create v0.1.0 --title "v0.1.0" --notes "..."
```

The workflow triggers on `release: types: [published]`. A draft release does not fire
it; the release has to actually be published.

### 5. Watch it land

The job does not start immediately. The `pypi` environment holds it for a ten-minute
wait timer, which exists so that a mistake noticed late can still be stopped — cancel
the run from the Actions tab and nothing reaches PyPI.

```console
$ gh run watch
$ pip index versions shirokumas
```

The workflow builds an sdist and a wheel with `python -m build` on Python 3.11 and
uploads them with `pypa/gh-action-pypi-publish`, which also attaches PEP 740
attestations — provenance PyPI can verify, and something trusted publishing is a
precondition for. The upload is recorded under the repository's Deployments tab.

## Things to know

**A version number cannot be reused.** PyPI rejects a re-upload of a version it already
has. Tagging the wrong commit and publishing means `0.1.0` is spent for good, and the
fix is to move on to `0.1.1` — deleting the release and the tag does not free the
number. Hence step 3.

**A failed publish does not spend the version number.** Only a successful upload
reserves it. If the trusted publisher configuration and the workflow disagree, the run
stops at authentication and PyPI never sees the release, so the fix is to correct the
configuration and re-run the failed job — the tag and the version survive. This is a
different situation from the one above, where the upload worked and the contents were
wrong.

**The `pypi` environment only accepts `v*` tags.** The workflow runs against the tag the
release points at, and the environment refuses anything else. A release created from a
branch, or from a tag named without the `v` prefix, will not publish.

**Both workflows check out with `fetch-depth: 0`.** The default shallow clone carries no
tags, which would leave every build claiming to be a development version. Do not drop
it.
