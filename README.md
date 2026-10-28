# unusual-activity-spotter

A command-line tool that finds unusual values in a CSV file, such as a transaction that
doesn't look like the rest of an account's activity.

It uses simple, explainable statistics (no machine learning), and it can judge each
account, customer, or category against **its own** normal rather than against everyone's.

Written in plain Python with no dependencies. The code is heavily commented so it can be
read top to bottom by someone who doesn't write code. The sample data is synthetic.

## Try it

```
python spotter.py sample_transactions.csv --value amount --by account
```

```
Checked 101 rows of 'amount' (each account judged against its own normal).
Flagged 4 unusual row(s) (cut-off: score beyond +/-3.5).

Line  account    amount   Typical   Score  Rest of row
----  --------  -------  --------  ------  ----------------------------------
  55  ACC-1001  1849.00     42.39  +136.3  2026-10-03 | Electronics Outlet
  87  ACC-1004  2300.00     99.16   +63.1  2026-10-19 | Wire Transfer Service
  70  ACC-1002   412.50     34.39   +44.0  2026-10-09 | Gift Cards Online
  80  ACC-1003    18.00  1,080.73    -7.0  2026-10-14 | Unknown Merchant

Not enough rows to judge (need at least 5): ACC-1005 (3 rows)

Skipped 1 row(s) with no usable number:
  line 74: no usable number in 'amount' ('n/a')
```

| Option | What it does |
| --- | --- |
| `--value COLUMN` | **Required.** The column holding the numbers to examine. |
| `--by COLUMN` | Judge each row against its own group, e.g. `--by account`. |
| `--threshold N` | How extreme a score must be to flag. Default 3.5. Higher means fewer flags. |
| `--min-group-size N` | Groups with fewer rows than this aren't judged. Default 5. |
| `--output FILE` | Also save the flagged rows to a new CSV, with their scores added. |

The tool only reads your file. It never changes it.

## Why `--by` matters

Run the same data without grouping:

```
python spotter.py sample_transactions.csv --value amount
```

This flags **16** rows. Twelve of them are just ACC-1003's ordinary rent, insurance, and
utility payments, which look huge next to everyone else's coffee and groceries. It also
**misses** the $18 charge on that account, the one thing that's actually odd for it.
Grouping fixes both problems, because "unusual" only means something relative to a
baseline.

## How it works

"Unusual" means "far from typical", so we need a measure of typical and a measure of
normal variation. The textbook choice is the average and the standard deviation, but
extreme values distort both:

```
Values: 10, 11, 12, 13, 950

Average: 199.2   Standard deviation: about 420
The 950 is only ~1.8 standard deviations out, under the usual cut-off of 3.
The outlier inflates the very yardstick that should catch it.
```

So the tool uses the **median** (the middle value) and the **MAD** (the median distance
from the middle). One extreme value barely moves either of them:

```
Median: 12   MAD: 1
The 950 scores about 630.
```

This is the "modified z-score" (Iglewicz & Hoaglin, 1993), and 3.5 is the cut-off that
paper recommends. A positive score means higher than typical, negative means lower.

## What to look for when reading the code

Everything is in [`spotter.py`](spotter.py), in the order the work happens:

- **`parse_number`** accepts `$1,234.50` the way bank exports write it, and refuses `nan`
  and `inf`, which Python would otherwise accept as numbers.
- **`load_records`** keeps going past bad rows and reports each one by line number.
- **`robust_scores`** is the heart of the tool. Note the fallback for when the MAD is zero
  (a subscription that always charges $9.99): the published method switches formulas
  rather than dividing by zero.
- **`find_unusual`** groups the rows, refuses to judge groups that are too small, and
  sorts the results most extreme first.
- **`format_table`** builds the aligned report you see above.

## Tests

```
python -m unittest -v
```

35 tests. [`test_spotter.py`](test_spotter.py) reads like a specification, and it includes
a test that proves the textbook method misses the example above while this one catches it.

## Limits

This is a first-pass screen, not a fraud model.

- It looks at **one number at a time**. It doesn't consider time of day, merchant,
  location, or the order of events.
- It assumes an account's normal behavior is fairly steady. Spending is often lopsided
  (many small purchases, a few big ones), so a legitimate large purchase can be flagged.
  Raise `--threshold` if you see too many of those.
- It needs a few rows of history per group to say anything at all.
- A flag means "worth a look", not "wrong".
