"""Tests for spotter.

Each test sets up a small, known situation and checks the tool's answer. Reading
them is a good way to see what the tool is supposed to do, and the first few
double as a worked explanation of the statistics.

Run them from this folder with:   python -m unittest -v
"""

import contextlib
import csv
import io
import statistics
import tempfile
import unittest
from pathlib import Path

import spotter
from spotter import Record

SAMPLE_FILE = Path(__file__).with_name("sample_transactions.csv")


def make_records(values, group="A", start_line=2):
    """Build in-memory records for a list of numbers, all in one group."""
    return [
        Record(start_line + i, {"account": group, "amount": str(v)}, float(v))
        for i, v in enumerate(values)
    ]


class StatisticsTests(unittest.TestCase):
    """The scoring maths, in isolation."""

    def test_robust_scores_on_a_known_example(self):
        # Median is 3. Distances from it are 2, 1, 0, 1, 97, so the MAD is 1.
        # Score = 0.6745 * (value - median) / MAD.
        median, scores = spotter.robust_scores([1, 2, 3, 4, 100])
        self.assertEqual(median, 3)
        expected = [-1.349, -0.6745, 0.0, 0.6745, 65.4265]
        for got, want in zip(scores, expected):
            self.assertAlmostEqual(got, want, places=4)

    def test_classic_method_misses_what_the_robust_method_catches(self):
        # This is the example from the top of spotter.py.
        numbers = [10, 11, 12, 13, 950]

        # The textbook "how many standard deviations from the average?" check:
        classic_z = (950 - statistics.fmean(numbers)) / statistics.stdev(numbers)
        self.assertLess(classic_z, 3)  # under the usual cut-off: NOT flagged

        # The median-based check:
        _, scores = spotter.robust_scores(numbers)
        self.assertGreater(scores[-1], 100)  # enormous: flagged

    def test_identical_values_are_never_unusual(self):
        _, scores = spotter.robust_scores([9.99] * 8)
        self.assertEqual(scores, [0.0] * 8)

    def test_fallback_when_most_values_are_identical(self):
        # Seven values, six identical: the MAD is zero, so the method must
        # switch to its fallback instead of dividing by zero.
        _, scores = spotter.robust_scores([5, 5, 5, 5, 5, 5, 9])
        self.assertTrue(all(s == 0 for s in scores[:-1]))
        self.assertGreater(scores[-1], spotter.DEFAULT_THRESHOLD)


class FindUnusualTests(unittest.TestCase):
    """Deciding which rows to flag."""

    def test_a_high_outlier_is_flagged(self):
        analysis = spotter.find_unusual(make_records([10, 12, 11, 13, 10, 12, 500]))
        self.assertEqual([f.record.value for f in analysis.flags], [500.0])
        self.assertGreater(analysis.flags[0].score, 0)

    def test_a_low_outlier_is_flagged_with_a_negative_score(self):
        analysis = spotter.find_unusual(make_records([100, 101, 99, 100, 102, 98, 100, 10]))
        self.assertEqual([f.record.value for f in analysis.flags], [10.0])
        self.assertLess(analysis.flags[0].score, 0)

    def test_ordinary_variation_is_not_flagged(self):
        analysis = spotter.find_unusual(make_records([20, 25, 22, 30, 27, 24, 29, 21]))
        self.assertEqual(analysis.flags, [])

    def test_threshold_controls_how_strict_it_is(self):
        records = make_records([10, 12, 11, 13, 10, 12, 500])
        self.assertEqual(len(spotter.find_unusual(records, threshold=3.5).flags), 1)
        self.assertEqual(len(spotter.find_unusual(records, threshold=10_000).flags), 0)

    def test_most_extreme_rows_come_first(self):
        records = make_records([10, 12, 11, 13, 10, 12, 500, 5000])
        values = [f.record.value for f in spotter.find_unusual(records).flags]
        self.assertEqual(values, [5000.0, 500.0])

    def test_judging_each_group_separately_avoids_false_alarms(self):
        # One account spends about $22 a time. Another pays about $1,000 a
        # time (think rent). Judged together, the second account looks wild.
        small = make_records([20 + i % 5 for i in range(30)], group="small")
        big = make_records([1000, 1010, 990, 1005, 995, 1002], group="big", start_line=100)
        records = small + big

        together = spotter.find_unusual(records)
        self.assertEqual({f.record.line_number for f in together.flags}, set(range(100, 106)))

        separately = spotter.find_unusual(records, group_column="account")
        self.assertEqual(separately.flags, [])

    def test_group_baseline_still_catches_a_real_oddity(self):
        big = make_records([1000, 1010, 990, 1005, 995, 1002, 18], group="big")
        analysis = spotter.find_unusual(big, group_column="account")
        self.assertEqual([f.record.value for f in analysis.flags], [18.0])

    def test_small_groups_are_reported_not_judged(self):
        records = make_records([10, 10, 5000], group="tiny")
        analysis = spotter.find_unusual(records, group_column="account")
        self.assertEqual(analysis.flags, [])
        self.assertEqual(analysis.small_groups, {"tiny": 3})

    def test_min_group_size_can_be_changed(self):
        records = make_records([10, 11, 10, 12, 5000])  # 5 rows
        self.assertEqual(len(spotter.find_unusual(records, min_group_size=5).flags), 1)
        self.assertEqual(len(spotter.find_unusual(records, min_group_size=6).flags), 0)

    def test_blank_group_cells_get_a_visible_label(self):
        records = [Record(2 + i, {"account": "", "amount": "1"}, 1.0) for i in range(3)]
        analysis = spotter.find_unusual(records, group_column="account")
        self.assertEqual(analysis.small_groups, {"(blank)": 3})


class ParseNumberTests(unittest.TestCase):
    def test_plain_and_formatted_numbers(self):
        self.assertEqual(spotter.parse_number("12.5"), 12.5)
        self.assertEqual(spotter.parse_number("  7 "), 7.0)
        self.assertEqual(spotter.parse_number("$1,234.50"), 1234.5)
        self.assertEqual(spotter.parse_number("-3"), -3.0)

    def test_unusable_text_gives_none(self):
        for text in ("", "n/a", "abc", "12abc"):
            with self.subTest(text=text):
                self.assertIsNone(spotter.parse_number(text))

    def test_nan_and_infinity_are_rejected(self):
        # Python would otherwise accept these words as numbers.
        for text in ("nan", "NaN", "inf", "-inf", "infinity"):
            with self.subTest(text=text):
                self.assertIsNone(spotter.parse_number(text))


class LoadingTests(unittest.TestCase):
    """Reading CSV files, including messy ones."""

    def csv_file(self, text, encoding="utf-8"):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "data.csv"
        path.write_bytes(text.encode(encoding) if isinstance(text, str) else text)
        return path

    def test_unusable_rows_are_skipped_and_reported_with_line_numbers(self):
        path = self.csv_file('date,amount\nd1,10\nd2,n/a\nd3,\nd4,"1,234.50"\n')
        data = spotter.load_records(path, "amount", None)
        self.assertEqual([r.value for r in data.records], [10.0, 1234.5])
        self.assertEqual(len(data.skipped), 2)
        self.assertIn("line 3", data.skipped[0])
        self.assertIn("line 4", data.skipped[1])

    def test_record_line_numbers_match_the_file(self):
        path = self.csv_file("date,amount\nd1,10\nd2,20\n")
        data = spotter.load_records(path, "amount", None)
        self.assertEqual([r.line_number for r in data.records], [2, 3])

    def test_short_and_long_rows_do_not_crash(self):
        path = self.csv_file("date,amount\nd1\nd2,5,surprise-extra-cell\n")
        data = spotter.load_records(path, "amount", None)
        self.assertEqual([r.value for r in data.records], [5.0])
        self.assertEqual(len(data.skipped), 1)  # the short row had no amount

    def test_excel_byte_order_mark_is_handled(self):
        path = self.csv_file(b"\xef\xbb\xbfdate,amount\nd,5\n")
        data = spotter.load_records(path, "amount", None)
        self.assertEqual(data.fieldnames, ["date", "amount"])

    def test_missing_value_column_explains_what_is_available(self):
        path = self.csv_file("date,amount\nd,5\n")
        with self.assertRaises(ValueError) as caught:
            spotter.load_records(path, "price", None)
        self.assertIn("price", str(caught.exception))
        self.assertIn("date, amount", str(caught.exception))

    def test_missing_group_column_is_an_error_too(self):
        path = self.csv_file("date,amount\nd,5\n")
        with self.assertRaises(ValueError):
            spotter.load_records(path, "amount", "account")

    def test_empty_file_is_an_error(self):
        path = self.csv_file("")
        with self.assertRaises(ValueError):
            spotter.load_records(path, "amount", None)


class ReportTests(unittest.TestCase):
    def analyse(self, text, **options):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "data.csv"
        path.write_text(text)
        dataset = spotter.load_records(path, "amount", options.get("group"))
        analysis = spotter.find_unusual(dataset.records, options.get("group"))
        return dataset, analysis

    def test_report_lists_flagged_rows_with_their_line_numbers(self):
        rows = "".join(f"d,{v}\n" for v in (10, 11, 12, 10, 11, 12, 900))
        dataset, analysis = self.analyse("date,amount\n" + rows)
        report = spotter.format_report(analysis, dataset, "amount", None, 3.5)
        self.assertIn("Flagged 1 unusual row(s)", report)
        # The flagged row (900) is on line 8 of the file: 1 header + 7 values.
        flagged_line = next(l for l in report.splitlines() if "900" in l)
        self.assertTrue(flagged_line.strip().startswith("8"))

    def test_report_says_so_when_nothing_is_unusual(self):
        rows = "".join(f"d,{v}\n" for v in (10, 11, 12, 10, 11, 12))
        dataset, analysis = self.analyse("date,amount\n" + rows)
        report = spotter.format_report(analysis, dataset, "amount", None, 3.5)
        self.assertIn("Nothing unusual found.", report)

    def test_report_mentions_skipped_rows_and_small_groups(self):
        text = "account,amount\nA,1\nA,2\nA,n/a\n"
        dataset, analysis = self.analyse(text, group="account")
        report = spotter.format_report(analysis, dataset, "amount", "account", 3.5)
        self.assertIn("Not enough rows to judge", report)
        self.assertIn("Skipped 1 row(s)", report)

    def test_saved_csv_has_original_columns_plus_scores(self):
        rows = "".join(f"d,{v}\n" for v in (10, 11, 12, 10, 11, 12, 900))
        dataset, analysis = self.analyse("date,amount\n" + rows)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out = Path(temp.name) / "flagged.csv"
        spotter.write_flagged_csv(out, analysis, dataset)

        with out.open(newline="") as handle:
            saved = list(csv.DictReader(handle))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["amount"], "900")
        self.assertIn("group_typical", saved[0])
        self.assertTrue(saved[0]["unusual_score"].startswith("+"))


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        """Run main() and capture what it prints."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = spotter.main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_sample_file_flags_exactly_the_four_planted_oddities(self):
        code, out, _ = self.run_cli(SAMPLE_FILE, "--value", "amount", "--by", "account")
        self.assertEqual(code, 0)
        self.assertIn("Flagged 4 unusual row(s)", out)
        for merchant in (
            "Electronics Outlet",
            "Gift Cards Online",
            "Unknown Merchant",
            "Wire Transfer Service",
        ):
            self.assertIn(merchant, out)

    def test_sample_file_without_grouping_gives_false_alarms(self):
        # This is the point of --by: without it, an account that normally pays
        # large amounts floods the results.
        _, grouped, _ = self.run_cli(SAMPLE_FILE, "--value", "amount", "--by", "account")
        _, together, _ = self.run_cli(SAMPLE_FILE, "--value", "amount")
        self.assertIn("Flagged 4 ", grouped)
        self.assertNotIn("Flagged 4 ", together)

    def test_missing_file(self):
        code, _, err = self.run_cli("nope.csv", "--value", "amount")
        self.assertEqual(code, 1)
        self.assertIn("Not a file", err)

    def test_wrong_column_name_lists_the_real_ones(self):
        code, _, err = self.run_cli(SAMPLE_FILE, "--value", "price")
        self.assertEqual(code, 1)
        self.assertIn("date, account, merchant, amount", err)

    def test_nonsense_threshold_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                spotter.main([str(SAMPLE_FILE), "--value", "amount", "--threshold", "0"])
        self.assertEqual(caught.exception.code, 2)

    def test_output_option_saves_the_flagged_rows(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out_path = Path(temp.name) / "flagged.csv"
        code, out, _ = self.run_cli(
            SAMPLE_FILE, "--value", "amount", "--by", "account", "--output", out_path
        )
        self.assertEqual(code, 0)
        self.assertIn("Saved 4 flagged row(s)", out)
        with out_path.open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_it_never_modifies_the_input_file(self):
        before = SAMPLE_FILE.read_bytes()
        self.run_cli(SAMPLE_FILE, "--value", "amount", "--by", "account")
        self.assertEqual(SAMPLE_FILE.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
