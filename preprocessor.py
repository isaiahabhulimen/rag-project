import re


def fix_spaced_text(line):
    tokens = line.split()

    if not tokens:
        return line

    single_char_count = sum(1 for token in tokens if len(token) == 1)
    single_char_ratio = single_char_count / len(tokens)

    if single_char_ratio > 0.7 and all(len(t) <= 2 for t in tokens):
        return "".join(tokens)

    return line


def clean_text(raw_text):
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }

    for unicode_char, ascii_char in replacements.items():
        raw_text = raw_text.replace(unicode_char, ascii_char)

    raw_text = re.sub(r"(\w)-\n(\w)", r"\1\2", raw_text)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

    paragraphs = raw_text.split("\n\n")
    cleaned_paragraphs = []

    for paragraph in paragraphs:

        lines = paragraph.split("\n")

        cleaned_lines = []

        for line in lines:
            line = fix_spaced_text(line)
            line = re.sub(r" {2,}", " ", line)
            line = line.strip()

            if line:
                cleaned_lines.append(line)

        cleaned_paragraphs.append(" ".join(cleaned_lines))

    return "\n\n".join(cleaned_paragraphs)
