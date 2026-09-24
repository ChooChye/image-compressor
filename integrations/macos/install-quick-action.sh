#!/bin/zsh
# Installs the Finder Quick Actions (right-click > Quick Actions):
#   Compress for Web          -> compress-for-web.sh   (AVIF + WebP via smartimg)
#   Compress (Keep Format)    -> compress-keep-format.sh (same format via smart_compressor.py)
# Re-run after editing either script to update the installed copies.
set -e
here=${0:A:h}
repo=${here:h:h}
python=$(command -v python3)

install_action() { # install_action <menu name> <script>
  local name=$1 dest="$HOME/Library/Services/$1.workflow" body
  body=$(<"$here/$2")
  body=${body//@PYTHON@/$python}
  body=${body//@COMPRESSOR@/$repo/smart_compressor.py}

  rm -rf "$dest"
  cp -R "$here/template.workflow" "$dest"
  plutil -replace NSServices.0.NSMenuItem.default -string "$name" "$dest/Contents/Info.plist"
  plutil -replace actions.0.action.ActionParameters.COMMAND_STRING -string "$body" \
    "$dest/Contents/document.wflow"

  # New Quick Actions start hidden; enable it in the Finder context menu, Quick Actions and Services.
  local key="(null) - $name - runWorkflowAsService" tmp=$(mktemp) pb=/usr/libexec/PlistBuddy
  defaults export pbs "$tmp"
  $pb -c "Print :NSServicesStatus" "$tmp" >/dev/null 2>&1 || $pb -c "Add :NSServicesStatus dict" "$tmp"
  $pb -c "Delete :NSServicesStatus:'$key'" "$tmp" 2>/dev/null || true
  $pb -c "Add :NSServicesStatus:'$key':presentation_modes dict" "$tmp"
  for mode in ContextMenu FinderPreview ServicesMenu TouchBar; do
    $pb -c "Add :NSServicesStatus:'$key':presentation_modes:$mode bool true" "$tmp"
  done
  defaults import pbs "$tmp"
  rm -f "$tmp"
  echo "Installed: $dest"
}

"$python" -c "import numpy, PIL" 2>/dev/null ||
  echo "warning: $python lacks numpy/Pillow; run: $python -m pip install -r $repo/requirements.txt"

install_action "Compress for Web" compress-for-web.sh
install_action "Compress (Keep Format)" compress-keep-format.sh
/System/Library/CoreServices/pbs -update

echo "Finder: right-click image(s) or a folder > Quick Actions"
