#!/bin/bash
set -e

PAGES_DIR="/Users/salinasd/Documents/code/geoalgo.github.io/lmarena-analysis"

mkdir -p "$PAGES_DIR"
cp lmarena-analysis/index.html "$PAGES_DIR/"
cp lmarena-analysis/dashboard_data.json "$PAGES_DIR/"
cp lmarena-analysis/org_metadata.json "$PAGES_DIR/"

cd "$PAGES_DIR"
git add index.html dashboard_data.json org_metadata.json
git commit -m "edit leaderboard"
git push
