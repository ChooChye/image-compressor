#!/bin/zsh
# Finder Quick Action body: compress the selected images and/or folders for the web.
#   folder  -> <folder>/compressed
#   images  -> <their folder>/compressed (images sharing a folder run as one batch,
#              so web-report.json and picture-snippets.html cover all of them)
# Output of every run is appended to ~/Library/Logs/smartimg-quick-action.log.

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
log="$HOME/Library/Logs/smartimg-quick-action.log"
opts=(--avif-quality 70 --max-height 1600)

notify() {
  osascript -e 'on run argv' -e 'display notification (item 2 of argv) with title (item 1 of argv)' -e 'end run' "$1" "$2"
}

if ! command -v smartimg >/dev/null; then
  notify "Compress for Web" "smartimg not found. Run: cargo install --path apps/cli"
  exit 1
fi

typeset -A files_by_dir
typeset -a out_dirs
failed=0

run() { # run <out-dir> <inputs...>
  local out=$1; shift
  print -r -- "=== $(date '+%F %T') smartimg web $* --out-dir $out" >>"$log"
  smartimg web "$@" --out-dir "$out" "${opts[@]}" >>"$log" 2>&1 || failed=1
  out_dirs+=("$out")
}

notify "Compress for Web" "Compressing $# item(s)..."

for p in "$@"; do
  if [[ -d $p ]]; then
    run "${p%/}/compressed" "$p"
  elif [[ -f $p ]]; then
    files_by_dir[${p:h}]+="$p"$'\n'
  fi
done

for dir in ${(k)files_by_dir}; do
  run "$dir/compressed" "${(@f)${files_by_dir[$dir]%$'\n'}}"
done

if (( failed )); then
  notify "Compress for Web" "Done with errors. See ~/Library/Logs/smartimg-quick-action.log"
  open -a Console "$log"
else
  notify "Compress for Web" "Done: ${#out_dirs} folder(s) written"
fi

# Reveal the first output folder in Finder.
[[ -d ${out_dirs[1]} ]] && open "${out_dirs[1]}"
