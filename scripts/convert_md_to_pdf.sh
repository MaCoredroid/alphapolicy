#!/usr/bin/env bash
set -euo pipefail

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found in PATH" >&2
  exit 1
fi
if ! command -v xelatex >/dev/null 2>&1; then
  echo "xelatex not found in PATH" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=(doc.md alpha.md leakage.md loss.md policy.md retro.md)
fi

declare -a TMP_FILES=()
cleanup() {
  local f
  if [ "${TMP_FILES+x}" != "x" ]; then
    return 0
  fi
  for f in "${TMP_FILES[@]}"; do
    if [ -n "$f" ] && [ -e "$f" ]; then
      rm -f "$f"
    fi
  done
}
trap cleanup EXIT

make_tmp() {
  local t
  t=$(mktemp)
  TMP_FILES+=("$t")
  printf "%s" "$t"
}

filter=$(make_tmp)
cat > "$filter" <<'LUA'
local replacements = {
  ["\u{2192}"] = "\\to",
  ["\u{21D2}"] = "\\Rightarrow",
  ["\u{2248}"] = "\\approx",
  ["\u{2264}"] = "\\le",
  ["\u{2265}"] = "\\ge",
  ["\u{2208}"] = "\\in",
  ["\u{221D}"] = "\\propto",
  ["\u{2260}"] = "\\neq",
  ["\u{03BC}"] = "\\mu",
  ["\u{03B1}"] = "\\alpha",
  ["\u{03B2}"] = "\\beta",
  ["\u{03C4}"] = "\\tau",
  ["\u{03BB}"] = "\\lambda",
  ["\u{03A3}"] = "\\Sigma",
  ["\u{00D7}"] = "\\times",
  ["\u{22C5}"] = "\\cdot",
}

function Str(el)
  local s = el.text
  local inlines = pandoc.Inlines{}
  local buffer = {}
  for _, codepoint in utf8.codes(s) do
    local ch = utf8.char(codepoint)
    if ch == "\u{2011}" then
      buffer[#buffer + 1] = "-"
    else
      local repl = replacements[ch]
      if repl then
        if #buffer > 0 then
          inlines:insert(pandoc.Str(table.concat(buffer)))
          buffer = {}
        end
        inlines:insert(pandoc.Math("InlineMath", repl))
      else
        buffer[#buffer + 1] = ch
      end
    end
  end
  if #buffer > 0 then
    inlines:insert(pandoc.Str(table.concat(buffer)))
  end
  if #inlines == 1 then
    return inlines[1]
  elseif #inlines > 0 then
    return inlines
  end
  return el
end
LUA

for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "missing file: $f" >&2
    exit 1
  fi

  tmp=$(make_tmp)
  python3 - "$f" "$tmp" <<'PY'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]

text_re = re.compile(r"\\text\{([^}]*)\}")

unicode_math_map = {
    "\u2192": r"\\to",
    "\u21d2": r"\\Rightarrow",
    "\u2248": r"\\approx",
    "\u2264": r"\\le",
    "\u2265": r"\\ge",
    "\u2208": r"\\in",
    "\u221d": r"\\propto",
    "\u2260": r"\\neq",
    "\u03bc": r"\\mu",
    "\u03b1": r"\\alpha",
    "\u03b2": r"\\beta",
    "\u03c4": r"\\tau",
    "\u03bb": r"\\lambda",
    "\u03a3": r"\\Sigma",
    "\u00d7": r"\\times",
    "\u22c5": r"\\cdot",
}


def replace_unicode_math(s: str) -> str:
    return "".join(unicode_math_map.get(ch, ch) for ch in s)


def fix_text_underscores(s: str) -> str:
    def repl(m):
        inner = m.group(1)
        inner = re.sub(r"(?<!\\)_", r"\\_", inner)
        return "\\text{" + inner + "}"
    return text_re.sub(repl, s)


in_code = False
in_math = False
block_indent = ""
block_lines = []

out = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.rstrip("\n")

        if stripped.strip().startswith("```"):
            in_code = not in_code

        if not in_code and stripped.strip() == "[" and not in_math:
            in_math = True
            block_indent = stripped[:len(stripped) - len(stripped.lstrip())]
            block_lines = []
            continue

        if in_math:
            if stripped.strip() == "]":
                if out and out[-1].strip() != "":
                    out.append("")
                out.append(block_indent + "$$")
                for bl in block_lines:
                    out.append(block_indent + bl)
                out.append(block_indent + "$$")
                out.append("")
                in_math = False
                block_indent = ""
                block_lines = []
                continue

            line_out = stripped
            if line_out.startswith(block_indent):
                line_out = line_out[len(block_indent):]
            line_out = line_out.replace("(\\mu\\cdot r)", "($\\mu\\cdot r$)")
            line_out = line_out.replace("\u2011", "-")
            line_out = replace_unicode_math(line_out)
            line_out = fix_text_underscores(line_out)
            if line_out.strip():
                block_lines.append(line_out)
            continue

        line_out = stripped
        if not in_code:
            line_out = line_out.replace("(\\mu\\cdot r)", "($\\mu\\cdot r$)")
            line_out = line_out.replace("\u2011", "-")
            line_out = fix_text_underscores(line_out)
        out.append(line_out)

with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
PY

  out="${f%.md}.pdf"
  pandoc --from markdown+tex_math_dollars+tex_math_single_backslash \
    --lua-filter="$filter" \
    --pdf-engine=xelatex \
    --variable=monofont=Menlo \
    "$tmp" -o "$out"

  echo "wrote $out"
done
