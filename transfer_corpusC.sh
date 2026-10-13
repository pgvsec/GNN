#!/bin/bash
# transfer_corpusC.sh — run this on Negishi to push the Corpus C cache to Gilbreth.
#
# Usage (from Negishi login node, after the cache build completes):
#   bash transfer_corpusC.sh
#
# What it does:
#   rsync the built cache directory from ~/python/corpusC/cache on Negishi
#   to /scratch/gilbreth/gupt1042/corpusC/cache on Gilbreth, then
#   validates the file count on the remote side.
#
# Prerequisites:
#   - SSH key already set up between Negishi and Gilbreth (both are Purdue RCAC;
#     your ~/.ssh/id_rsa is shared across clusters via AFS home).
#   - Cache built: ~/python/corpusC/cache/meta.json must exist.
#     Build it with:
#       python -c "from data_c import build_cache_c; \
#           build_cache_c('$HOME/python/corpusC', '$HOME/python/corpusC/cache')"

set -e

SRC="$HOME/python/corpusC/cache"
DST_HOST="gilbreth.rcac.purdue.edu"
DST_PATH="/scratch/gilbreth/gupt1042/corpusC/cache"
DST="${DST_HOST}:${DST_PATH}"

# Sanity-check that the cache looks complete before transferring.
if [ ! -f "$SRC/meta.json" ]; then
    echo "ERROR: $SRC/meta.json not found — cache may not have built yet."
    echo "Run: python -c \"from data_c import build_cache_c; build_cache_c('$HOME/python/corpusC', '$HOME/python/corpusC/cache')\""
    exit 1
fi

LOCAL_COUNT=$(ls "$SRC"/*.npy 2>/dev/null | wc -l)
echo "Local cache: $LOCAL_COUNT .npy files + meta.json"
echo "Transferring to $DST ..."

# Create the destination directory on Gilbreth first.
ssh "$DST_HOST" "mkdir -p $DST_PATH"

# rsync with compression; -P shows progress and resumes partial transfers.
rsync -avzP --stats "$SRC/" "$DST/"

# Validate remote file count matches local.
REMOTE_COUNT=$(ssh "$DST_HOST" "ls $DST_PATH/*.npy 2>/dev/null | wc -l")
echo ""
echo "Local .npy count:  $LOCAL_COUNT"
echo "Remote .npy count: $REMOTE_COUNT"

if [ "$LOCAL_COUNT" -eq "$REMOTE_COUNT" ]; then
    echo "Transfer complete and validated."
    echo ""
    echo "Next steps (on Gilbreth):"
    echo "  cd /scratch/gilbreth/gupt1042/GNN && git pull origin claude/eloquent-cerf-l0zqbo"
    echo "  sbatch train_layer3.slurm"
else
    echo "WARNING: file counts differ — re-run this script to resume the transfer."
    exit 1
fi
