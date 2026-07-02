# Publishing `causal-certificate`

The package is packaging-clean and builds/installs (verified). Publishing to PyPI is
**not** done — it is an irreversible, public action (the name is claimed permanently and
a version can never be re-uploaded). Do it deliberately, and run the upload **yourself**
so your API token never lands in a chat transcript or shell history you don't control.

## Recommended: zero-token automated publishing (already wired)

`.github/workflows/publish.yml` publishes to PyPI via **Trusted Publishing** (OIDC) when
you cut a GitHub Release — **no API token exists anywhere**, so there is nothing to leak.
One-time enable after the repo is on GitHub:
1. PyPI → https://pypi.org/manage/account/publishing/ → add a *pending publisher*:
   owner `Akhilesh-Gogikar`, repo `causal-certificate`, workflow `publish.yml`.
   Leave **Environment name blank** — this workflow does not use a GitHub Actions
   environment (kept simple: fewer fields that have to match exactly).
2. Bump `version` in `pyproject.toml`, then cut a GitHub Release → CI builds and publishes.

`.github/workflows/ci.yml` runs the certificate test + `twine check` on every push/PR.
The manual `twine upload` path below is the fallback if you skip CI.

## 0. Decide first
- **Name.** `causal-certificate` looked free at last check, but it sits in a crowded
  causal-*inference* namespace (causal-learn, causalml, causallib). A more distinctive
  distribution name (e.g. `strictly-causal`, `causal-cert`) may read better. The *import*
  name can stay `causal_certificate` regardless — only `name = ...` in `pyproject.toml`
  is the PyPI id.
- **Timing.** Consider waiting until the companion paper is on arXiv so the README can
  link it; or ship GitHub-only first (`pip install git+https://github.com/.../...`),
  which claims no permanent name.

## 1. Build (no secrets)
```bash
cd causal_certificate
python -m pip install --upgrade build twine
python -m build            # -> dist/causal_certificate-0.1.0{.tar.gz,-py3-none-any.whl}
python -m twine check dist/*
```

## 2. Dry-run on TestPyPI (recommended)
Get a token at https://test.pypi.org/manage/account/token/ — do NOT paste it here.
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-testpypi-token>     # in YOUR shell only
python -m twine upload --repository testpypi dist/*
# verify in a throwaway venv:
python -m venv /tmp/cc && /tmp/cc/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ causal-certificate
/tmp/cc/bin/python -c "import causal_certificate; print(causal_certificate.__version__)"
```

## 3. Real upload (irreversible)
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-pypi-token>          # scope it to this project after first upload
python -m twine upload dist/*
```

## Notes
- Bump `version` in `pyproject.toml` for every upload — PyPI rejects re-uploading a version.
- After the first upload, create a **project-scoped** token and delete the account-wide one.
- Prefer PyPI **Trusted Publishing** (OIDC from GitHub Actions) over a long-lived token if
  you set up CI — then no token exists to leak.
