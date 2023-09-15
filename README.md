# Kernel Patches Daemon

Kernel Patches Daemon (`kpd`), watches patchwork for new series sent to the mailing
list, applies the series on top of `repo`.

`repo` maintains base branches by pulling updates from `<upstream>@<upstream_branch>`
and applying the CI files from `<ci_repo>@<ci_branch>` to it.

When `kpd` sees a new or updated series, it applies the patches from the series
on top of one of the maintained branch and creates a PR against it.

This in turns triggers the Github workflows copied from `ci_repo`.

When the workflow runs are reporting back, `kpd` updates the relevant checks for
this series on patchwork (when the configuration provides a `pw_token` and
`pw_user`).

## Configuration
`kpd.conf.template` is an example config based on the setup of https://github.com/kernel-patches/bpf

The branch `bpf-next` uses [Github personal token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
to authenticate, while the branch `bpf`
uses [Github App](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/about-authentication-with-a-github-app).

When using a GH app, it needs to have the following read and write access:
- Content (write to repo)
- Pull request (create PRs)
- Workflow

## Building
```
# Install poetry
pip install --user poetry

# Setup virtualenv
python -m venv .venv
poetry install

# Test
poetry run python -m unittests
```

## Running
```
poetry run python -m kernel_patches_daemon --config <config_path> --label-color configs/labels.json
```

## Docker

Kernel Patches Daemon is available as pre-build image:

```
$ docker pull ghcr.io/facebookincubator/kernel-patches-daemon:latest
```

To build Kernel Patches Daemon with [Docker](https://docs.docker.com/engine/install):

1. Install [Docker](https://docs.docker.com/engine/install) and [docker-compose](https://docs.docker.com/compose/install/)
2. Build image
```
$ docker-compose build
```
3. Start KPD service
```
$ docker-compose up
```

## CONTRIBUTING
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
Kernel Patches Daemon is BSD licensed, as found in the LICENSE file.
