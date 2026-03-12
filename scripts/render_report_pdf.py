import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "docs" / "reports"
SRC = REPORTS_DIR / "report_lsq_cuda_cn.md"
OUT = REPORTS_DIR / "report_lsq_cuda_cn.pdf"
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


def normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            lines.append(" | ".join(cells))
            continue
        if not line:
            lines.append("")
            continue
        indent = ""
        if line.startswith("### "):
            lines.append(line)
            continue
        if line.startswith("## "):
            lines.append(line)
            continue
        if line.startswith("# "):
            lines.append(line)
            continue
        if line.startswith("- "):
            indent = "  "
        wrapped = textwrap.wrap(
            line,
            width=44,
            break_long_words=False,
            replace_whitespace=False,
            subsequent_indent=indent,
        )
        lines.extend(wrapped or [""])
    return lines


def render_pdf(lines: list[str], out_path: Path) -> None:
    page_width, page_height = 8.27, 11.69
    top = 0.96
    bottom = 0.06
    line_h = 0.022
    usable = top - bottom
    lines_per_page = math.floor(usable / line_h)
    font_prop = font_manager.FontProperties(fname=str(FONT_PATH))

    with PdfPages(out_path) as pdf:
        for start in range(0, len(lines), lines_per_page):
            fig = plt.figure(figsize=(page_width, page_height))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = top
            for line in lines[start : start + lines_per_page]:
                if line.startswith("# "):
                    ax.text(0.06, y, line[2:], fontsize=18, fontweight="bold", fontproperties=font_prop)
                    y -= line_h * 1.6
                elif line.startswith("## "):
                    ax.text(0.06, y, line[3:], fontsize=14, fontweight="bold", fontproperties=font_prop)
                    y -= line_h * 1.35
                elif line.startswith("### "):
                    ax.text(0.06, y, line[4:], fontsize=12, fontweight="bold", fontproperties=font_prop)
                    y -= line_h * 1.2
                else:
                    ax.text(0.06, y, line, fontsize=10.5, fontproperties=font_prop)
                    y -= line_h
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = normalize_lines(text)
    render_pdf(lines, OUT)
    print(f"PDF written to {OUT}")


if __name__ == "__main__":
    main()
