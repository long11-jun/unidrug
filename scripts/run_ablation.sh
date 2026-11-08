#!/bin/bash
# Component ablation from the paper (Tables VI-VIII).
# A  LLM baseline          no_cmsa,no_cmon
# B  baseline + CMSA       no_cmon
# C  baseline + CMON       no_cmsa
# D  full minus 2-hop      cmsa_no_2hop
# E  full minus digest     cmon_no_digest
# F  full minus LODC       cmon_no_lodc
# G  full model
set -e

TASK="${1:-}"
if [ -z "${TASK}" ]; then
    echo "Usage: $0 <dp|ddi|ppi> [dataset] [split]"
    echo "  dp  SIDER|ClinTox|HIV|Bace"
    echo "  ddi Luo"
    echo "  ppi SHS27k|SHS148k  BFS|DFS"
    exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGS=("no_cmsa,no_cmon" "no_cmon" "no_cmsa" "cmsa_no_2hop" "cmon_no_digest" "cmon_no_lodc" "")
NAMES=("A_baseline" "B_cmsa" "C_cmon" "D_no_2hop" "E_no_digest" "F_no_lodc" "G_full")

for i in "${!FLAGS[@]}"; do
    FLAG="${FLAGS[$i]}"
    NAME="${NAMES[$i]}"
    echo "${NAME} ${FLAG}"
    case "${TASK}" in
        dp)
            DATASET="${2:-SIDER}"
            "${DIR}/run_dp.sh" "${DATASET}" single "" "${FLAG}"
            ;;
        ddi)
            "${DIR}/run_ddi.sh" LUO single "${FLAG}"
            ;;
        ppi)
            DATASET="${2:-SHS27k}"
            SPLIT="${3:-DFS}"
            "${DIR}/run_ppi.sh" "${DATASET}" "${SPLIT}" single "" "0" "${FLAG}"
            ;;
        *)
            echo "Ablation in the paper is DP, DDI, and PPI."
            exit 1
            ;;
    esac
done
