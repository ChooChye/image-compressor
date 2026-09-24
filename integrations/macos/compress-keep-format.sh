#!/bin/zsh
# Finder Quick Action body: compress the selected images and/or folders, keeping each
# image's format (JPEG stays JPEG, PNG stays PNG, ...) via smart_compressor.py.
#   folder  -> <folder>/compressed/<same name>   (subfolders not included)
#   image   -> <its folder>/compressed/<same name>
# The original is copied unchanged when compression can't make it smaller.
# Output of every run is appended to ~/Library/Logs/smartimg-quick-action.log.
# @PYTHON@ and @COMPRESSOR@ are filled in by install-quick-action.sh.

setopt extendedglob
zmodload zsh/parameter
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
python="@PYTHON@"
compressor="@COMPRESSOR@"
log="$HOME/Library/Logs/smartimg-quick-action.log"
target=0.99        # minimum SSIM; lower (e.g. 0.985) = smaller files, slightly softer
max_dimension=2560 # longest side in pixels; 0 = never resize
jobs_max=4

notify() {
  osascript -e 'on run argv' -e 'display notification (item 2 of argv) with title (item 1 of argv)' -e 'end run' "$1" "$2"
}

if [[ ! -f $compressor ]]; then
  notify "Compress (Keep Format)" "smart_compressor.py not found. Re-run install-quick-action.sh"
  exit 1
fi

typeset -a images out_dirs
supported='*.(#i)(jpg|jpeg|png|webp|avif)'
skipped=0
for p in "$@"; do
  if [[ -d $p ]]; then
    images+=("${p%/}"/${~supported}(N.))
  elif [[ -f $p && ${p:t} == ${~supported} ]]; then
    images+=("$p")
  else
    (( skipped++ ))
  fi
done

if (( ! ${#images} )); then
  notify "Compress (Keep Format)" "No JPEG, PNG, WebP or AVIF images selected"
  exit 0
fi

notify "Compress (Keep Format)" "Compressing ${#images} image(s)..."
failures=$(mktemp)

compress_one() {
  local src=$1 out_dir=${1:h}/compressed format=${1:e:l}
  [[ $format == jpg ]] && format=jpeg
  mkdir -p "$out_dir"
  local result
  if result=$("$python" "$compressor" "$src" -o "$out_dir/${src:t}" --formats $format \
      --target $target --max-dimension $max_dimension 2>&1); then
    print -r -- "OK $src"$'\n'"$result" >>"$log"
  else
    print -r -- "FAILED $src"$'\n'"$result" >>"$log"
    print -r -- "$src" >>"$failures"
  fi
}

print -r -- "=== $(date '+%F %T') compress keep format: ${#images} image(s)" >>"$log"
for img in "${images[@]}"; do
  while (( ${#jobstates} >= jobs_max )); do sleep 0.2; done
  compress_one "$img" &
  out_dirs+=("${img:h}/compressed")
done
wait

failed=$(wc -l <"$failures" | tr -d ' ')
rm -f "$failures"
if (( failed )); then
  notify "Compress (Keep Format)" "$failed of ${#images} failed. See ~/Library/Logs/smartimg-quick-action.log"
  open -a Console "$log"
else
  msg="Done: ${#images} image(s)"
  (( skipped )) && msg+=", $skipped unsupported skipped"
  notify "Compress (Keep Format)" "$msg"
fi

# Reveal the first output folder in Finder.
[[ -d ${out_dirs[1]} ]] && open "${out_dirs[1]}"
