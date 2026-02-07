"""Convert relative image paths in Markdown files to GitHub raw URLs."""

import os
import re
import sys


def build_raw_url(raw_base: str, file_dir: str, rel_path: str) -> str:
    """Resolve a relative image path to a raw.githubusercontent.com URL."""
    resolved = os.path.relpath(
        os.path.normpath(os.path.join(file_dir, rel_path)), "."
    ).replace(os.sep, "/")
    return f"{raw_base}/{resolved}"


def convert_file(fpath: str, raw_base: str) -> bool:
    """Replace relative image links in a single Markdown file.

    Returns True if any replacement was made.
    """
    img_re = re.compile(r'(!\[[^\]]*\]\()([^)\s]+)(\s*(?:"[^"]*")?\s*\))')

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    file_dir = os.path.dirname(fpath)

    def _replace(m: re.Match, _dir: str = file_dir) -> str:
        prefix, path, suffix = m.group(1), m.group(2), m.group(3)
        if path.startswith(("http://", "https://", "//")):
            return m.group(0)
        return f"{prefix}{build_raw_url(raw_base, _dir, path)}{suffix}"

    new_content = img_re.sub(_replace, content)

    if new_content != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: convert_image_paths.py <repository> <branch>")
        sys.exit(1)

    repository = sys.argv[1]
    branch = sys.argv[2]
    raw_base = f"https://raw.githubusercontent.com/{repository}/{branch}"

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(root, fname)
            if convert_file(fpath, raw_base):
                print(f"Converted: {fpath}")


if __name__ == "__main__":
    main()
