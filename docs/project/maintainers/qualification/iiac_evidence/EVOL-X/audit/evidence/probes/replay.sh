#!/bin/bash
# Independent Session C replay (read-only audit; worktrees only)
set -u
REPO=/d/Projekty/intergrax
OUT=$REPO/build/iiac-c/logs; mkdir -p $OUT
TES=3c4a641b3e48a13f437e830a0562e2a80b03aabb
FINAL=bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1
cd $REPO
git worktree add --detach build/iiac-c/final $FINAL > $OUT/wt_final.log 2>&1
git worktree add --detach build/iiac-c/tes $TES > $OUT/wt_tes.log 2>&1
unset VIRTUAL_ENV
( cd build/iiac-c/final && git rev-parse HEAD > $OUT/final_head.txt && uv sync --extra dev-ci --frozen > $OUT/sync_final.log 2>&1 \
  && uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish > $OUT/mode_a.log 2>&1; echo "MODE_A_EXIT=$?" >> $OUT/mode_a.log ) &
( cd build/iiac-c/tes && git rev-parse HEAD > $OUT/tes_head.txt && uv sync --extra dev-ci --frozen > $OUT/sync_tes.log 2>&1 \
  && uv run python scripts/qualification/evol_x_vl3_manifest.py --check > $OUT/vl3.log 2>&1; echo "VL3_EXIT=$?" >> $OUT/vl3.log; \
  uv run pytest tests/qualification/evol_x -q --basetemp=$REPO/build/iiac-c/pytmp --junitxml=$OUT/mode_b_junit.xml > $OUT/mode_b_pytest.log 2>&1; echo "PYTEST_EXIT=$?" >> $OUT/mode_b_pytest.log ) &
wait
echo ALL_DONE > $OUT/done.flag
