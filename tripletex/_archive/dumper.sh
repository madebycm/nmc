#!/bin/bash
# Watch task_log.jsonl on VPS for new solve2 entries and save each prompt to a local JSON file
DIR="$(dirname "$0")/logs/prompts_dump"
mkdir -p "$DIR"
echo "Watching for new prompts... (saving to $DIR)"
ssh vps "tail -n0 -f /opt/tripletex/task_log.jsonl" | while IFS= read -r line; do
    ts=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('logged_at','unknown'))" 2>/dev/null)
    fn="${ts//[:]/-}.json"
    echo "$line" | python3 -m json.tool > "$DIR/$fn" 2>/dev/null && echo "Saved: $fn"
done
