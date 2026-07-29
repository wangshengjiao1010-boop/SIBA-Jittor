#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-/root/autodl-tmp/datasets/downloads}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-/root/autodl-tmp/datasets/official}
M3FD_SHA256=${M3FD_SHA256:-ec33d031bbd26697b75061972786526cdd815ee8111586813427d155ec522dfc}

mkdir -p "$OFFICIAL_ROOT" "$OFFICIAL_ROOT/m3fd_extract" "$OFFICIAL_ROOT/official_tno/ir" "$OFFICIAL_ROOT/official_tno/vi"

m3fd_archive=$(find "$DOWNLOAD_ROOT/m3fd_tardal" -type f -name 'M3FD_Fusion.zip' -print -quit)
if [ -z "$m3fd_archive" ]; then
  echo "M3FD_Fusion.zip was not found under $DOWNLOAD_ROOT/m3fd_tardal" >&2
  exit 1
fi
echo "$M3FD_SHA256  $m3fd_archive" | sha256sum --check -
unzip -q -o "$m3fd_archive" -d "$OFFICIAL_ROOT/m3fd_extract"
m3fd_ir=$(find "$OFFICIAL_ROOT/m3fd_extract" -type d -iname 'Ir' -print -quit)
if [ -z "$m3fd_ir" ] || [ ! -d "$(dirname "$m3fd_ir")/Vis" ]; then
  echo "M3FD Ir/Vis directories were not found after extraction" >&2
  exit 1
fi
mkdir -p "$OFFICIAL_ROOT/Ir" "$OFFICIAL_ROOT/Vis"
cp -a "$m3fd_ir/." "$OFFICIAL_ROOT/Ir/"
cp -a "$(dirname "$m3fd_ir")/Vis/." "$OFFICIAL_ROOT/Vis/"

tno_ir=''
while IFS= read -r candidate; do
  candidate_vi="$(dirname "$candidate")/vi"
  if [ -d "$candidate_vi" ] && [ "$(find "$candidate" -maxdepth 1 -type f | wc -l)" -eq 45 ] && [ "$(find "$candidate_vi" -maxdepth 1 -type f | wc -l)" -eq 45 ]; then
    tno_ir=$candidate
    break
  fi
done < <(find "$DOWNLOAD_ROOT/tno_siba" -type d -iname 'ir' | sort)
if [ -z "$tno_ir" ]; then
  echo "A complete 45-pair TNO ir/vi directory was not found" >&2
  exit 1
fi
cp -a "$tno_ir/." "$OFFICIAL_ROOT/official_tno/ir/"
cp -a "$(dirname "$tno_ir")/vi/." "$OFFICIAL_ROOT/official_tno/vi/"

test "$(find "$OFFICIAL_ROOT/Ir" -maxdepth 1 -type f | wc -l)" -eq 300
test "$(find "$OFFICIAL_ROOT/Vis" -maxdepth 1 -type f | wc -l)" -eq 300
test "$(find "$OFFICIAL_ROOT/official_tno/ir" -maxdepth 1 -type f | wc -l)" -eq 45
test "$(find "$OFFICIAL_ROOT/official_tno/vi" -maxdepth 1 -type f | wc -l)" -eq 45

echo "Official M3FD and TNO sources are ready under $OFFICIAL_ROOT"
