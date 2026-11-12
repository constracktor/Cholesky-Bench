#!/usr/bin/env bash
# Add HPX 2.0.0 to the local Spack hpx recipe until it is available upstream.
# Safe to run repeatedly; does nothing if the recipe already has the version.
set -euo pipefail

pkg="$(spack location -p hpx)/package.py"
if grep -q 'version("2.0.0"' "$pkg"; then
  echo "HPX 2.0.0 already in $pkg"
  exit 0
fi
sed -i '/^    version("master", branch="master")$/a\    version("2.0.0", sha256="c755ab893e9fdcbc896c6eb735ad411cc78b9314e916d87119c3cd5304928211")' "$pkg"
grep -q 'version("2.0.0"' "$pkg" || {
  echo "Failed to add HPX 2.0.0 to $pkg" >&2
  exit 1
}
echo "Added HPX 2.0.0 to $pkg"
