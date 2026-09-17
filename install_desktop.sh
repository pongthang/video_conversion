#!/usr/bin/env bash
# Add the application to the desktop menu for the current user.
#
#   ./install_desktop.sh            install
#   ./install_desktop.sh --remove   uninstall
#
# Everything goes under ~/.local, so no root is needed and nothing outside the
# user's home is touched.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ID="miko-video-translator"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
DESKTOP_FILE="$DESKTOP_DIR/$APP_ID.desktop"
ICON_FILE="$ICON_DIR/$APP_ID.png"

GREEN=$'\033[32m'; BLUE=$'\033[34m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
info() { printf '%s==>%s %s\n' "$BLUE$BOLD" "$OFF" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$GREEN" "$OFF" "$*"; }

if [[ "${1:-}" == "--remove" ]]; then
  rm -f "$DESKTOP_FILE" "$ICON_FILE"
  command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
  ok "Removed the menu entry."
  exit 0
fi

[[ -x "$PROJECT_DIR/run_gui.sh" ]] || { echo "run_gui.sh missing" >&2; exit 1; }

info "Installing the menu entry for $USER"
mkdir -p "$DESKTOP_DIR" "$ICON_DIR"

SOURCE_ICON="$PROJECT_DIR/src/vtrans_gui/assets/app.png"
if [[ -f "$SOURCE_ICON" ]]; then
  cp "$SOURCE_ICON" "$ICON_FILE"
  ok "icon -> $ICON_FILE"
  ICON_VALUE="$APP_ID"
else
  ICON_VALUE="video-x-generic"
fi

cat > "$DESKTOP_FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=Miko Video Translator
GenericName=Video Dubbing
Comment=Turn Chinese video into English-dubbed, English-subtitled video
Exec=$PROJECT_DIR/run_gui.sh
Path=$PROJECT_DIR
Icon=$ICON_VALUE
Terminal=false
Categories=AudioVideo;Video;AudioVideoEditing;
Keywords=video;translate;subtitle;dub;chinese;english;whisper;
StartupNotify=true
StartupWMClass=vtrans_gui
DESKTOP

chmod +x "$DESKTOP_FILE"
ok "entry -> $DESKTOP_FILE"

command -v update-desktop-database >/dev/null 2>&1 && \
  update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && \
  gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

cat <<DONE

${GREEN}${BOLD}Installed.${OFF}

  Look for "Miko Video Translator" in the applications menu,
  or start it directly with:

    ./run_gui.sh

  To remove it again:  ./install_desktop.sh --remove
DONE
