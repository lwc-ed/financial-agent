#!/usr/bin/env bash
# statistical_test 完整流程
# 用法：cd ml_ibm/statistical_test && bash run_all.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"
LOG_FILE="$SCRIPT_DIR/logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "📝 Log 儲存至：$LOG_FILE"

run_step() {
    local step="$1"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  ▶  $step"
    echo "════════════════════════════════════════════════════"
    python "$step"
}

run_step pair_t_test.py
run_step pair_t_test_baseline.py
run_step wilcoxon_test.py
run_step wilcoxon_test_baseline.py

echo ""
echo "🎉  statistical_test 全部完成！"