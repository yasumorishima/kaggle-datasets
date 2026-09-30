#!/bin/bash
# Collect per-pitch ABS challenges from MLB StatsAPI game feeds (keyless).
# Usage: collect.sh START_DATE END_DATE_EXCLUSIVE   e.g. 2026-03-25 2026-09-28 (the full 2026 regular season)
# A game is marked done only when its feed downloaded (HTTP 2xx) and has plays,
# so a failed fetch is retried on the next run instead of being dropped silently.
set -u
cd "$(dirname "$0")"
mkdir -p games
out=challenges_all.jsonl
d=$1
while [ "$d" != "$2" ]; do
  for pk in $(curl -f --max-time 30 -sS "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&date=$d" \
      | jq -r '.dates[]?.games[] | select(.status.abstractGameState=="Final") | .gamePk'); do
    echo "$pk" >> scheduled.txt
    [ -f "games/$pk.done" ] && continue
    curl -f --max-time 60 -sS "https://statsapi.mlb.com/api/v1.1/game/$pk/feed/live" -o /tmp/absg.json || { echo "FETCH FAIL $pk" >&2; continue; }
    n=$(jq '.liveData.plays.allPlays | length' /tmp/absg.json)
    [ "${n:-0}" -gt 0 ] || { echo "NO PLAYS $pk" >&2; continue; }
    jq -c --arg date "$d" -f extract2.jq /tmp/absg.json >> "$out" && touch "games/$pk.done"
    sleep 1
  done
  d=$(date -I -d "$d + 1 day")
done
echo "DONE $(wc -l < "$out") rows, $(ls games | wc -l) games, $(sort -u scheduled.txt | wc -l) scheduled"
