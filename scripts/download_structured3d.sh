#!/usr/bin/env bash
# Download the Structured3D subset used by Structured3DFloor. Run this on the
# cluster login node (compute nodes are offline), preferably inside tmux/screen
# or via nohup. The default 00/01/02 slice is roughly 15 GB and is enough to
# build and validate the loader before fetching all indices 00..17.

# ------------------------------- CONFIG -----------------------------------
DEST="${DEST:-/home/ul/ul_student/ul_fnm03/data/Structured3D}"
PANO_IDX="${PANO_IDX:-00 01 02}"
GET_ANNOT="${GET_ANNOT:-1}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"
# --------------------------------------------------------------------------

BASE_URL="https://zju-kjl-jointlab-azure.kujiale.com/Structured3D"
ZIP_DIR="$DEST/zips"
DATASET_ROOT="$DEST/Structured3D"
MARKER_DIR="$DEST/.extracted"
mkdir -p "$ZIP_DIR" "$MARKER_DIR"

warn() { printf 'WARNING: %s\n' "$*" >&2; }

archive_payload_complete() {
    archive="$1"
    found=0
    while IFS= read -r payload; do
        found=1
        [ -f "$DEST/$payload" ] || return 1
    done < <(unzip -Z1 "$archive" 2>/dev/null | grep -E \
        '^Structured3D/scene_[^/]+/(annotation_3d\.json|2D_rendering/.*/panorama/.*/rgb_rawlight\.png)$' || true)
    [ "$found" -eq 1 ]
}

download_and_extract() {
    filename="$1"
    marker="$2"
    zip_path="$ZIP_DIR/$filename"

    if [ -f "$marker" ]; then
        printf '[skip] %s already extracted (%s)\n' "$filename" "$marker"
        return 0
    fi

    # If a retained archive is present, verify all relevant payload files before
    # downloading again. This also recovers idempotency after a lost marker.
    if [ -f "$zip_path" ] && archive_payload_complete "$zip_path"; then
        printf '[skip] %s payload already present; recording marker\n' "$filename"
        : > "$marker"
        [ "$KEEP_ZIPS" = "1" ] || rm -f -- "$zip_path"
        return 0
    fi

    printf '[download] %s\n' "$filename"
    if ! wget -c "$BASE_URL/$filename" -O "$zip_path"; then
        warn "download failed for $filename; partial file kept for wget -c"
        return 1
    fi

    printf '[extract] %s -> %s\n' "$filename" "$DEST"
    if ! unzip -o "$zip_path" -d "$DEST"; then
        warn "unzip failed for $filename; archive kept for inspection/resume"
        return 1
    fi
    : > "$marker"
    if [ "$KEEP_ZIPS" = "0" ]; then
        rm -f -- "$zip_path"
    fi
    return 0
}

if [ "$GET_ANNOT" = "1" ]; then
    download_and_extract "Structured3D_annotation_3d.zip" \
        "$MARKER_DIR/annotation_3d.done" || true
fi

for nn in $PANO_IDX; do
    case "$nn" in
        [01][0-9]) ;;
        *) warn "ignoring invalid panorama index '$nn' (expected 00..17)"; continue ;;
    esac
    if [ "$nn" -gt 17 ]; then
        warn "ignoring panorama index '$nn' (full set is 00..17)"
        continue
    fi
    download_and_extract "Structured3D_panorama_${nn}.zip" \
        "$MARKER_DIR/panorama_${nn}.done" || true
done

if [ -d "$DATASET_ROOT" ]; then
    scene_count="$(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'scene_*' | wc -l)"
    printf '\nScenes present: %s\n' "$scene_count"
    du -sh "$DATASET_ROOT" 2>/dev/null || warn "could not compute disk usage"
else
    warn "no extracted dataset root exists yet"
fi
printf 'Dataset root: %s\n' "$DATASET_ROOT"
printf 'Pass one scene as: --dataset structured3d --home %s/scene_XXXXX --config full\n' "$DATASET_ROOT"
