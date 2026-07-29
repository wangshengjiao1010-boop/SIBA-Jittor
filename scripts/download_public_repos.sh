#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/root/autodl-tmp/datasets/sources}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}
mkdir -p "$SOURCE_ROOT"
cd "$SOURCE_ROOT"

clone_or_update() {
  local url=$1
  local name=$2
  local commit=$3
  if [ -d "$name/.git" ]; then
    git -C "$name" fetch --depth 1 origin "$commit"
  else
    git clone --no-checkout "$url" "$name"
    git -C "$name" fetch --depth 1 origin "$commit"
  fi
  git -C "$name" checkout --detach "$commit"
}

clone_or_update https://github.com/Linfeng-Tang/MSRS.git MSRS 5fa66e33c5ba875eec4de4f5a5bf609bdcc566c5
clone_or_update https://github.com/hanna-xu/RoadScene.git RoadScene ab71420f8fc96396eeb97eecf3694909b3feb656
clone_or_update https://github.com/JinyuanLiu-CV/TarDAL.git TarDAL 6a9edd744b44fc03344fe8fb0fd930f5df47b00b
clone_or_update https://github.com/Linfeng-Tang/Evaluation-for-Image-Fusion.git Evaluation-for-Image-Fusion f5f055bcadb49c22fb734c3498aef6c56fc71f2a

mkdir -p "$PROJECT_ROOT/third_party"
evaluation_link="$PROJECT_ROOT/third_party/Evaluation-for-Image-Fusion"
if [ -L "$evaluation_link" ]; then
  ln -sfn "$SOURCE_ROOT/Evaluation-for-Image-Fusion" "$evaluation_link"
elif [ ! -e "$evaluation_link" ]; then
  ln -s "$SOURCE_ROOT/Evaluation-for-Image-Fusion" "$evaluation_link"
fi

find MSRS RoadScene TarDAL -maxdepth 3 -type d | sort
