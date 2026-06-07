#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="$ROOT/data"

echo "=== Cleaning generated files ==="

rm -vf "$DATA"/mtgdb.fs
rm -vf "$DATA"/mtgdb.fs.index
rm -vf "$DATA"/mtgdb.fs.lock
rm -vf "$DATA"/mtgdb.fs.tmp
rm -vf "$DATA"/mtgdb.fs.old

rm -vf "$DATA"/synergy_model.pt
rm -vf "$DATA"/card_projector.pt

rm -vf "$DATA"/chroma.sqlite3
find "$DATA" -maxdepth 1 -type d | while IFS= read -r d; do
    name=$(basename "$d")
    if [[ "$name" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
        rm -rvf "$d"
    fi
done

echo ""
echo "=== Importing collection (CSV + WCC decks) ==="
cd "$ROOT"
python import_collection.py

echo ""
echo "=== Building ChromaDB vector index ==="
python VectorDB.py

echo ""
echo "=== Training synergy model (Phase C) ==="
python ai/train.py

echo ""
echo "=== Done ==="
echo "Generated files:"
ls -lh "$DATA"/synergy_model.pt "$DATA"/card_projector.pt "$DATA"/mtgdb.fs "$DATA"/chroma.sqlite3 2>/dev/null
