# Releasing

Publishing a release to [PyPI](https://pypi.org/project/shirokumas/) means creating a
GitHub Release. The `Upload Python Package` workflow takes it from there.

This document is for maintainers. It needs push access to the repository and the
`PYPI_API_TOKEN` repository secret.

## Where the version comes from

The version is derived from the Git tag by
[setuptools-scm](https://setuptools-scm.readthedocs.io/), and from nothing else. There
is no version to edit in `pyproject.toml`, and `shirokumas.__version__` reads back the
metadata the build produced rather than being the thing that feeds it.

| state of the checkout | reported version |
| --- | --- |
| exactly on `v0.1.0`, clean tree | `0.1.0` |
| a few commits past it | `0.1.1.dev3+g1a2b3c4` |
| no reachable tag | `0.1.dev58+g1a2b3c4` |

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

```console
$ gh run watch
$ pip index versions shirokumas
```

The workflow builds an sdist and a wheel with `python -m build` on Python 3.11 and
uploads them with `pypa/gh-action-pypi-publish`.

## Things to know

**A version number cannot be reused.** PyPI rejects a re-upload of a version it already
has. Tagging the wrong commit and publishing means `0.1.0` is spent for good, and the
fix is to move on to `0.1.1` — deleting the release and the tag does not free the
number. Hence step 3.

**Both workflows check out with `fetch-depth: 0`.** The default shallow clone carries no
tags, which would leave every build claiming to be a development version. Do not drop
it.

**The tags `v0.0.1`..`v0.0.4` are not reachable from `main`.** They point into a history
that was later rewritten and share no common ancestor with `main`, which is why an
untagged `main` reports `0.1.devN` rather than `0.0.5.devN`. It has no effect on
releases: the published version depends only on the tag the release is cut from, so a
new tag produces exactly the version it names. The old tags do not need to be repaired.
