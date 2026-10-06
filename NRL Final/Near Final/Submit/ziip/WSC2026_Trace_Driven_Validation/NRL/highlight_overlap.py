"""Highlight text in NRL_main.tex that is identical to the conference paper.

Usage:  python highlight_overlap.py [N]
Reads NRL_main.tex and con417s3-file1.pdf, writes NRL_main_colored.tex.
Any run of >= N consecutive words (default 6) that also appears verbatim in the
conference paper is colored. Math, citation keys, labels, and comments are
ignored when comparing; color groups never cross braces, environments, display
math, table cells, or paragraph breaks, so the output compiles like the input.
"""
import re
import sys
import unicodedata

import pymupdf

TEX_IN = "NRL_main.tex"
TEX_OUT = "NRL_main_colored.tex"
PDF = "con417s3-file1.pdf"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
COLOR_RGB = "0,0,255"  # change to e.g. 200,0,0 for red

INLINE_ARG_CMDS = {"cite", "citep", "citet", "citeN", "citealp", "citeauthor",
                   "citeyear", "ref", "eqref", "label"}
SKIP_ARG_CMDS = {"includegraphics", "bibliography", "bibliographystyle", "url",
                 "usepackage", "color", "definecolor", "vspace", "hspace",
                 "pagestyle", "thispagestyle", "setlength", "input", "include",
                 "section", "subsection", "subsubsection", "paragraph",
                 "title", "author", "date"}
INLINE_CMDS = {"textit", "emph", "textbf", "texttt", "textsc", "ldots", "dots"}
MATH_ENVS = {"equation", "equation*", "align", "align*", "eqnarray",
             "eqnarray*", "gather", "gather*", "multline", "multline*",
             "displaymath", "math"}
ENVS_WITH_ARG = {"subfigure", "minipage", "tabular", "tabular*"}


def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())


# ---------------------------------------------------------------- PDF side
def pdf_tokens(path):
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"-\n", "-", text)                       # line-end hyphens
    text = re.sub(r"\([^()]{0,200}?\b(?:19|20)\d{2}[a-z]?\)", " ", text)  # (Author 2024)
    toks = [norm(t) for t in re.findall(r"[A-Za-z0-9'\u2019-]+", text)]
    return [t for t in toks if t]


# ---------------------------------------------------------------- TeX side
def skip_group(s, i, open_ch="{", close_ch="}"):
    """s[i] == open_ch; return index just past the matching close_ch."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def skip_args(s, i, n_opt=2, n_req=1):
    j = skip_ws(s, i)
    if j < len(s) and s[j] == "*":
        i = j = j + 1
    for _ in range(n_opt):
        j = skip_ws(s, i)
        if j < len(s) and s[j] == "[":
            i = skip_group(s, j, "[", "]")
    for _ in range(n_req):
        j = skip_ws(s, i)
        if j < len(s) and s[j] == "{":
            i = skip_group(s, j)
    return i


def is_word_char(c):
    return c.isalnum() or c == "'"


def tex_tokens(s, start, stop):
    """Return list of (start, end, normalized word, context id)."""
    toks, ctx, i = [], 0, start
    while i < stop:
        c = s[i]
        if c == "%":
            j = s.find("\n", i)
            i = stop if j < 0 else j
        elif c == "\\":
            m = re.match(r"\\([A-Za-z]+)(\*?)", s[i:])
            if m:
                name = m.group(1)
                i += m.end()
                if name in ("begin", "end"):
                    ctx += 1
                    j = skip_ws(s, i)
                    env_end = skip_group(s, j)
                    env = s[j + 1:env_end - 1]
                    i = env_end
                    if name == "begin" and env in MATH_ENVS:
                        k = s.find("\\end{" + env + "}", i)
                        i = stop if k < 0 else k + len(env) + 6
                    elif name == "begin":
                        i = skip_args(s, i, n_opt=1,
                                      n_req=1 if env in ENVS_WITH_ARG else 0)
                elif name in INLINE_ARG_CMDS:
                    i = skip_args(s, i)
                elif name in SKIP_ARG_CMDS:
                    ctx += 1
                    i = skip_args(s, i)
                elif name not in INLINE_CMDS:
                    ctx += 1
            else:
                nxt = s[i + 1] if i + 1 < stop else ""
                if nxt == "[":
                    k = s.find("\\]", i + 2)
                    i = stop if k < 0 else k + 2
                    ctx += 1
                elif nxt == "(":
                    k = s.find("\\)", i + 2)
                    i = stop if k < 0 else k + 2
                else:
                    if nxt == "\\":
                        ctx += 1
                    i += 2
        elif c == "$":
            if s.startswith("$$", i):
                k = s.find("$$", i + 2)
                i = stop if k < 0 else k + 2
                ctx += 1
            else:
                k = i + 1
                while k < stop and not (s[k] == "$" and s[k - 1] != "\\"):
                    k += 1
                i = k + 1
        elif c in "{}&":
            ctx += 1
            i += 1
        elif c == "\n":
            if re.match(r"\n[ \t]*\n", s[i:i + 50]):
                ctx += 1
            i += 1
        elif is_word_char(c):
            j = i
            while j < stop and (is_word_char(s[j]) or (
                    s[j] == "-" and j + 1 < stop and is_word_char(s[j + 1])
                    and is_word_char(s[j - 1]))):
                j += 1
            w = norm(s[i:j])
            if w:
                toks.append((i, j, w, ctx))
            i = j
        else:
            i += 1
    return toks


def main():
    s = open(TEX_IN, encoding="utf-8").read()
    body_start = s.index("\\begin{document}") + len("\\begin{document}")
    body_stop = s.index("\\end{document}")

    ref = pdf_tokens(PDF)
    grams = {tuple(ref[k:k + N]) for k in range(len(ref) - N + 1)}

    toks = tex_tokens(s, body_start, body_stop)
    words = [t[2] for t in toks]
    marked = [False] * len(toks)
    for k in range(len(words) - N + 1):
        if tuple(words[k:k + N]) in grams:
            for m in range(k, k + N):
                marked[m] = True

    # Group consecutive marked tokens that share a context into segments.
    segments, k = [], 0
    while k < len(toks):
        if not marked[k]:
            k += 1
            continue
        m = k
        while m + 1 < len(toks) and marked[m + 1] and toks[m + 1][3] == toks[k][3]:
            m += 1
        segments.append((toks[k][0], toks[m][1]))
        k = m + 1

    out = s
    for a, b in reversed(segments):
        out = out[:b] + "}" + out[b:]
        out = out[:a] + "{\\color{confoverlap}" + out[a:]

    preamble = ("\\usepackage{xcolor}\n"
                "\\definecolor{confoverlap}{RGB}{" + COLOR_RGB + "}\n")
    out = out.replace("\\usepackage{graphicx}",
                      "\\usepackage{graphicx}\n" + preamble, 1)
    note = ("\n\\noindent\\fbox{\\parbox{\\dimexpr\\linewidth-2\\fboxsep-2\\fboxrule}"
            "{\\small Text shown in {\\color{confoverlap}this color} is identical "
            "to the conference version of this work (Proceedings of the 2026 "
            "Winter Simulation Conference).}}\n")
    out = out.replace("\\maketitle", "\\maketitle" + note, 1)

    open(TEX_OUT, "w", encoding="utf-8").write(out)
    n_marked = sum(marked)
    print(f"N={N}: {n_marked}/{len(toks)} words "
          f"({100 * n_marked / len(toks):.1f}%) highlighted in "
          f"{len(segments)} segments -> {TEX_OUT}")


if __name__ == "__main__":
    main()
