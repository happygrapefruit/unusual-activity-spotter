#!/usr/bin/env python3
"""
spotter: find unusual values in a CSV file.

WHAT IT DOES
------------
Give it a spreadsheet-style (CSV) file and tell it which column holds the
numbers to examine, such as a transaction amount. It lists the rows whose value
looks out of the ordinary, and how far out of the ordinary each one is.

Optionally you can also name a column to group by, such as an account. Then each
row is judged against that account's OWN normal behavior rather than against
everyone's. That matters: a $1,100 payment is routine for one account and
alarming for another.

THE IDEA, IN PLAIN ENGLISH
--------------------------
"Unusual" has to mean "far from typical", so we need a measure of "typical" and
a measure of "how much values normally vary". The classic textbook choice is the
average and the standard deviation. It has a well-known weakness, though: the
unusual values themselves distort both numbers.

    Values: 10, 11, 12, 13, 950

    Average: 199.2.  Standard deviation: about 420.
    The 950 is only ~1.8 standard deviations from the average. The classic
    method calls that unremarkable (a common cut-off is 3), so the outlier
    hides itself by inflating the very yardstick used to catch it.

So this tool uses the MEDIAN (the middle value when sorted) and the MAD (the
median of how far each value is from that middle). The median barely moves when
one extreme value shows up, so the yardstick stays honest:

    Median: 12.  MAD: 1.  The 950 scores about 630. Impossible to miss.

This is called a "modified z-score" (Iglewicz & Hoaglin, 1993). The usual
cut-off for "unusual" is a score beyond +/-3.5, which is this tool's default.

USAGE
-----
    python spotter.py sample_transactions.csv --value amount --by account
    python spotter.py data.csv --value amount --threshold 5
    python spotter.py data.csv --value amount --by account --output flagged.csv
"""

# Lets us write modern type hints (like list[str]) on older Python versions.
# It changes nothing about how the program behaves.
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# A score beyond this (in either direction) is flagged. 3.5 is the value
# recommended in the original research on this method.
DEFAULT_THRESHOLD = 3.5

# A group needs at least this many rows before we'll judge it. With only two or
# three data points, "typical" isn't meaningful, and we'd rather say "not enough
# history" than flag things at random.
DEFAULT_MIN_GROUP_SIZE = 5

# Scaling constants from the published method. The first makes the MAD
# comparable to a standard deviation for bell-curve-shaped data. The second is
# the equivalent for the fallback used when the MAD is zero (explained below).
MAD_SCALE = 0.6745
MEAN_AD_SCALE = 1.253314

# The group name used when the user doesn't ask to group rows at all.
ALL_ROWS = "(all rows)"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
# A "dataclass" is a lightweight way to bundle related values under one name,
# so the code can say `flag.score` instead of juggling tuples and positions.


@dataclass
class Record:
    """One usable row from the file."""

    line_number: int  # where it sits in the file (the header row is line 1)
    fields: dict[str, str]  # the whole row exactly as read, by column name
    value: float  # the number we're examining, already converted


@dataclass
class Dataset:
    """Everything read from the file."""

    fieldnames: list[str]  # column names, in file order
    records: list[Record]  # rows that had a usable number
    skipped: list[str]  # explanations for rows we had to skip


@dataclass
class Flag:
    """One row judged to be unusual."""

    record: Record
    group: str  # which group it was compared within
    typical: float  # the group's median (its "typical" value)
    score: float  # how far from typical; + means high, - means low


@dataclass
class Analysis:
    """The outcome of judging a whole dataset."""

    flags: list[Flag]
    rows_checked: int
    # Groups with too few rows to judge: {group name: number of rows}.
    small_groups: dict[str, int]


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def parse_number(text: str) -> float | None:
    """Turn text like '$1,234.50' into 1234.5, or return None if it isn't a number.

    Bank exports often include currency signs and thousands separators, so we
    strip those. Anything else unreadable returns None, and the caller decides
    what to do about it.
    """
    cleaned = text.strip().replace("$", "").replace(",", "").replace(" ", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    # Python happily reads the words "nan" and "inf" as numbers. They would
    # poison every calculation, so we refuse them.
    if not math.isfinite(number):
        return None
    return number


def load_records(path: Path, value_column: str, group_column: str | None) -> Dataset:
    """Read the CSV file, keeping rows that have a usable number.

    Raises ValueError (with a plain-English message) if the file is empty or
    a requested column doesn't exist.
    """
    # "utf-8-sig" quietly handles the invisible marker that Excel sometimes
    # puts at the very start of CSV files.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError("the file is empty or has no header row")
        fieldnames = list(reader.fieldnames)

        for wanted in (value_column, group_column):
            if wanted and wanted not in fieldnames:
                raise ValueError(
                    f"there is no column named {wanted!r}. "
                    f"Columns in this file: {', '.join(fieldnames)}"
                )

        records: list[Record] = []
        skipped: list[str] = []

        # start=2 because line 1 is the header, so the first data row is line 2.
        for line_number, row in enumerate(reader, start=2):
            raw = row.get(value_column) or ""
            value = parse_number(raw)
            if value is None:
                skipped.append(
                    f"line {line_number}: no usable number in {value_column!r} ({raw!r})"
                )
                continue

            # Tidy the row: turn missing cells (None) into empty text, and drop
            # any stray extra cells from rows with too many commas.
            fields = {key: (cell or "") for key, cell in row.items() if key is not None}
            records.append(Record(line_number, fields, value))

    return Dataset(fieldnames, records, skipped)


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------


def robust_scores(numbers: list[float]) -> tuple[float, list[float]]:
    """Score each number by how far it sits from the group's typical value.

    Returns (median, scores), with one score per number, in the same order.
    A score of 0 means "exactly typical". Positive means higher than typical,
    negative means lower. Scores beyond about +/-3.5 are conventionally "unusual".
    """
    median = statistics.median(numbers)

    # How far is each value from the median, ignoring direction?
    distances = [abs(x - median) for x in numbers]

    # The MAD ("median absolute deviation"): the typical distance from typical.
    mad = statistics.median(distances)

    if mad > 0:
        # The standard method.
        scores = [MAD_SCALE * (x - median) / mad for x in numbers]
    else:
        # The MAD is zero when more than half the values are identical (for
        # example, a subscription that always charges $9.99). Dividing by zero
        # is impossible, so the published method switches to the mean absolute
        # deviation instead, which still notices the few values that differ.
        mean_distance = statistics.fmean(distances)
        if mean_distance == 0:
            # Every value is identical, so nothing can be unusual.
            scores = [0.0] * len(numbers)
        else:
            scores = [(x - median) / (MEAN_AD_SCALE * mean_distance) for x in numbers]

    return median, scores


def find_unusual(
    records: list[Record],
    group_column: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> Analysis:
    """Judge every record against its group's normal range and flag the outliers."""
    # Step 1: sort the records into groups (or one big group, if not grouping).
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if group_column:
            # A blank group cell gets a visible label rather than an empty name.
            name = record.fields.get(group_column, "").strip() or "(blank)"
        else:
            name = ALL_ROWS
        groups[name].append(record)

    flags: list[Flag] = []
    small_groups: dict[str, int] = {}
    rows_checked = 0

    # Step 2: score each group separately, so every group is held to its own
    # standard of normal.
    for name, members in groups.items():
        if len(members) < min_group_size:
            small_groups[name] = len(members)
            continue

        rows_checked += len(members)
        median, scores = robust_scores([m.value for m in members])
        for member, score in zip(members, scores):
            if abs(score) > threshold:
                flags.append(Flag(member, name, median, score))

    # Step 3: put the most extreme rows first. Ties go in file order.
    flags.sort(key=lambda f: (-abs(f.score), f.record.line_number))
    return Analysis(flags, rows_checked, small_groups)


# ---------------------------------------------------------------------------
# Presenting results
# ---------------------------------------------------------------------------


def format_report(
    analysis: Analysis,
    dataset: Dataset,
    value_column: str,
    group_column: str | None,
    threshold: float,
) -> str:
    """Build the human-readable report as one block of text."""
    scope = (
        f"each {group_column} judged against its own normal"
        if group_column
        else "all rows judged together"
    )
    lines = [
        f"Checked {analysis.rows_checked:,} rows of {value_column!r} ({scope}).",
        f"Flagged {len(analysis.flags)} unusual row(s) (cut-off: score beyond +/-{threshold:g}).",
    ]

    if analysis.flags:
        lines.append("")
        lines.extend(format_table(analysis, value_column, group_column))
        lines.append("")
        lines.append(
            "Typical = the group's middle value. Score = how far the row is from "
            "typical (+ higher, - lower)."
        )
    else:
        lines.append("Nothing unusual found.")

    if analysis.small_groups:
        names = ", ".join(f"{n} ({c} rows)" for n, c in sorted(analysis.small_groups.items()))
        lines.append("")
        lines.append(f"Not enough rows to judge (need at least 5): {names}")

    if dataset.skipped:
        lines.append("")
        lines.append(f"Skipped {len(dataset.skipped)} row(s) with no usable number:")
        lines.extend(f"  {reason}" for reason in dataset.skipped)

    return "\n".join(lines)


def format_table(
    analysis: Analysis, value_column: str, group_column: str | None
) -> list[str]:
    """Lay the flagged rows out as an aligned text table."""
    # Which columns count as "the rest of the row"? Everything except the ones
    # already shown in their own table column.
    shown_elsewhere = {value_column, group_column}
    other_columns = [
        name for name in analysis.flags[0].record.fields if name not in shown_elsewhere
    ]

    headers = ["Line"]
    if group_column:
        headers.append(group_column)
    headers += [value_column, "Typical", "Score", "Rest of row"]

    table = []
    for flag in analysis.flags:
        cells = [str(flag.record.line_number)]
        if group_column:
            cells.append(flag.group)
        cells += [
            flag.record.fields[value_column],
            f"{flag.typical:,.2f}",
            f"{flag.score:+.1f}",
            " | ".join(flag.record.fields[name] for name in other_columns),
        ]
        table.append(cells)

    # Make every column as wide as its longest entry, so everything lines up.
    widths = [max(len(row[i]) for row in [headers] + table) for i in range(len(headers))]

    def render(cells: list[str]) -> str:
        # Left-align text columns; right-align the numbers.
        parts = []
        for index, (cell, width) in enumerate(zip(cells, widths)):
            is_text = headers[index] in (group_column, "Rest of row")
            parts.append(cell.ljust(width) if is_text else cell.rjust(width))
        return "  ".join(parts).rstrip()

    return [render(headers), render(["-" * w for w in widths])] + [render(r) for r in table]


def write_flagged_csv(path: Path, analysis: Analysis, dataset: Dataset) -> None:
    """Save the flagged rows to a new CSV: the original columns plus two more."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(dataset.fieldnames + ["group_typical", "unusual_score"])
        for flag in analysis.flags:
            original = [flag.record.fields.get(name, "") for name in dataset.fieldnames]
            writer.writerow(original + [f"{flag.typical:.2f}", f"{flag.score:+.2f}"])


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, run the analysis, print the report.

    Returns 0 on success and 1 on a problem with the input. (The number
    returned is the program's "exit code", which other tools can check.)
    """
    parser = argparse.ArgumentParser(
        description="Find unusual values in a CSV file using robust statistics."
    )
    parser.add_argument("csv_file", type=Path, help="the CSV file to examine")
    parser.add_argument(
        "--value", required=True, metavar="COLUMN", help="column holding the numbers to examine"
    )
    parser.add_argument(
        "--by",
        metavar="COLUMN",
        help="judge each row against its own group, e.g. --by account",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"how extreme a score must be to flag (default: {DEFAULT_THRESHOLD:g}; "
        "higher = fewer flags)",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        metavar="N",
        help=f"groups with fewer rows are not judged (default: {DEFAULT_MIN_GROUP_SIZE})",
    )
    parser.add_argument(
        "--output", type=Path, metavar="FILE", help="also save the flagged rows to this CSV file"
    )
    args = parser.parse_args(argv)

    # parser.error() prints a friendly message and exits with code 2.
    if args.threshold <= 0:
        parser.error("--threshold must be greater than 0")
    if args.min_group_size < 3:
        parser.error("--min-group-size must be at least 3")

    if not args.csv_file.is_file():
        print(f"Not a file: {args.csv_file}", file=sys.stderr)
        return 1

    try:
        dataset = load_records(args.csv_file, args.value, args.by)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    analysis = find_unusual(dataset.records, args.by, args.threshold, args.min_group_size)
    print(format_report(analysis, dataset, args.value, args.by, args.threshold))

    if args.output:
        write_flagged_csv(args.output, analysis, dataset)
        print(f"\nSaved {len(analysis.flags)} flagged row(s) to {args.output}")

    return 0


# This guard means "only run main() when the file is executed directly", not
# when another file (like the tests) merely imports it.
if __name__ == "__main__":
    sys.exit(main())
