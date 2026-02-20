#!/bin/bash
# 다운로드 완료 대기 → 추출 → 검증 자동 실행
set -e
DATA_DIR="/home/node/.openclaw/workspace/pdm-agent/data"
ZIP="$DATA_DIR/ims_bearing_dataset.zip"
IMS_DIR="$DATA_DIR/ims"
export PATH="/tmp:$PATH"

echo "=== 다운로드 완료 대기 중 ==="
while true; do
    SIZE=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
    # 1.08GB = ~1,080,000,000
    if [ "$SIZE" -gt 1070000000 ]; then
        # wget이 아직 쓰고 있을 수 있으니 2초 대기 후 재확인
        sleep 2
        SIZE2=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
        if [ "$SIZE" -eq "$SIZE2" ]; then
            echo "다운로드 완료: $(du -h $ZIP | cut -f1)"
            break
        fi
    fi
    sleep 10
    echo "  대기 중... $(du -h $ZIP 2>/dev/null | cut -f1)"
done

echo ""
echo "=== 추출 중 ==="
rm -rf "$IMS_DIR"
cd /home/node/.openclaw/workspace/pdm-agent
python3 scripts/download_ims_dataset.py --keep-zip 2>&1

echo ""
echo "=== Part 3a 검증 실행 ==="
python3 scripts/validate_part3a.py --interval 10 2>&1
