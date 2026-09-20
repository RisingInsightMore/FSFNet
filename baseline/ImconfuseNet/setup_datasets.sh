#!/bin/bash
# setup_datasets.sh — 在服务器上创建数据集符号链接
# 用法: bash setup_datasets.sh
# 前提: 在 baseline/ImconfuseNet/ 目录下执行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
TARGET_BASE="${SCRIPT_DIR}/../../Datasets"

DATASETS=("AVIID" "DayDrone" "NightDrone")

mkdir -p "${DATASETS_DIR}"

for ds in "${DATASETS[@]}"; do
    TARGET="${TARGET_BASE}/${ds}"
    LINK="${DATASETS_DIR}/${ds}"

    if [ ! -d "${TARGET}" ]; then
        echo "[WARN] 数据集目录不存在: ${TARGET}"
        continue
    fi

    if [ -L "${LINK}" ]; then
        echo "[SKIP] 符号链接已存在: ${LINK} -> $(readlink "${LINK}")"
    elif [ -d "${LINK}" ]; then
        echo "[WARN] 目录已存在(非符号链接): ${LINK}，跳过"
    else
        ln -s "${TARGET}" "${LINK}"
        echo "[OK] 创建符号链接: ${LINK} -> ${TARGET}"
    fi
done

echo ""
echo "验证数据集链接:"
for ds in "${DATASETS[@]}"; do
    LINK="${DATASETS_DIR}/${ds}"
    if [ -L "${LINK}" ] && [ -d "${LINK}" ]; then
        echo "  [OK] ${ds}: $(ls "${LINK}" | tr '\n' ' ')"
    else
        echo "  [FAIL] ${ds}: 链接无效或不存在"
    fi
done
