from typing import Iterable, Generator
from collections import Counter
from pathlib import Path

import time
import numpy.random as npr


RNG = npr.default_rng(0)

def tuplesToFile(path: Path, generator: Iterable[tuple], sep: str="\t", lastsep: str="\t"):
    path.parent.mkdir(exist_ok=True, parents=True)
    time.sleep(0.5)
    print("Writing", path.as_posix(), "...")
    time.sleep(0.5)
    with open(path, "w", encoding="utf-8") as handle:
        for tup in generator:
            handle.write(sep.join(map(str,tup[:-2])) + sep + lastsep.join(map(str,tup[-2:])) + "\n")
    return path


def countTsvValues(path: Path, column: int) -> Counter:
    frequencies = Counter()
    for row in iterateTsv(path):
        frequencies[row[column]] += 1
    return frequencies


def iterateTsv(path: Path) -> Generator[tuple, None, None]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            yield tuple(line.strip().split("\t"))


def limitTsvValueCount(path: Path, column: int, max_frequency: int):
    """
    In the given column, find values that appear more than the max_frequency. Then sample max_frequency of its examples
    and ditch the rest.
    """
    value_counts = countTsvValues(path, column)
    values_to_be_reduced = {key for key,count in value_counts.items() if count > max_frequency}
    rows_to_keep = {value: set(RNG.choice(value_counts[value], size=max_frequency, replace=False))
        for value in values_to_be_reduced
    }

    def filteredTuples() -> Generator[tuple, None, None]:
        value_specific_enumerate = Counter()
        for row in iterateTsv(path):
            value = row[column]
            if value not in values_to_be_reduced or value_specific_enumerate[value] in rows_to_keep[value]:
                yield row
            value_specific_enumerate[value] += 1

    out_path = path.with_stem(path.stem + f"_limited-{max_frequency}")
    return tuplesToFile(
        out_path,
        filteredTuples()
    )


def histogramOfTsv(path: Path, column: int):
    """
    Makes a histogram of the ith value on each line of a TSV file.
    """
    from fiject import MultiHistogram
    from tqdm.auto import tqdm

    h = MultiHistogram(f"counts-{path.stem}")
    with open(path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle):
            parts = line.strip().split("\t")
            h.add(f"column {column}", float(parts[column]))

    h.commitWithArgs_histplot(
        MultiHistogram.ArgsGlobal(binwidth=1, relative_counts=True, center_ticks=True, x_label="Label", y_label="Fraction of examples")
    )
