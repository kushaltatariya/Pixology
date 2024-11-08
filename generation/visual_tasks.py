from typing import Iterable, Union
from collections import Counter
from pathlib import Path

import itertools
import numpy as np
import numpy.random as npr
import wonderwords as ww
from tqdm.auto import tqdm
from string import ascii_lowercase

from generation.tsv import tuplesToFile

RNG = npr.default_rng(0)

PATH_EXAMPLES = Path(__file__).resolve().parent
PATH_ROOT = PATH_EXAMPLES.parent
PATH_DATA = PATH_ROOT / "data"


def generateSentences(loop: bool):
    while True:
        with open(PATH_DATA / "probing" / "sentence_length.txt", "r", encoding="utf-8") as handle:
            for line in handle:
                split, _, sentence = line.strip().split("\t")
                yield sentence

        if loop:
            print("End of sentences reached. Recommencing...")
        else:
            break


def generateWords(lengths: Union[Iterable[int],int]):
    generator = ww.RandomWord()

    if isinstance(lengths, int):
        lengths = [lengths]
    length_iterator = itertools.cycle(lengths)

    while True:
        yield " ".join(generator.random_words(amount=next(length_iterator)))


def generateCharacters(lengths: Union[Iterable[int], int]):
    if isinstance(lengths, int):
        lengths = [lengths]
    length_iterator = itertools.cycle(lengths)

    characters = ascii_lowercase + " "
    while True:
        length = next(length_iterator)
        yield "".join([characters[i] for i in RNG.choice(len(characters), size=length)])


##################################################################################################################


def dataset_unicodeInsert(generator: Iterable[str]):
    # 50% should be with and 50% without
    character_ords = list(range(1568,1610+1)) + list(range(1654,1725+1))  # Arabic characters that aren't just diacritics.
    for string in generator:
        if RNG.random() < 0.5:
            yield 0, string
        else:
            index = RNG.choice(len(string))
            character_ord = RNG.choice(character_ords)
            yield 1, string[:index] + chr(character_ord) + string[index:]


def dataset_countCharacter(generator: Iterable[str], examples_per_sentence: int):
    def normalised_softmax(x: np.ndarray, tau: float):
        # Normalise
        x = x / np.sum(x)

        # Softmax (first apply temperature, then apply invariant shift)
        x = x / tau
        x = np.exp(x - min(x))
        return x / np.sum(x)

    for string in generator:
        counts = Counter(string)
        for k in list(counts.keys()):
            if not k.isalpha():
                counts.pop(k)

        if len(counts) < 3:  # Sentence consists of spacing/numbers/punctuation, or very few letters, such that counting characters becomes counting length.
            continue

        keys          = list(counts.keys())
        probabilities = normalised_softmax(np.array(list(counts.values())), tau=0.05)  # Lower temperature (towards 0) makes the sample less uniform and more likely to sample the argmax, which helps to avoid 1-count samples.
        for index in RNG.choice(len(keys), p=probabilities, size=min(examples_per_sentence, len(keys)), replace=False):
            yield counts[keys[index]], string, keys[index]


def dataset_maxCharacter(generator: Iterable[str]):
    for string in generator:
        counts = Counter(string)
        for k in list(counts.keys()):
            if not k.isalpha():
                counts.pop(k)

        if len(counts) < 3:  # Sentence consists of spacing/numbers/punctuation, or very few letters, such that counting characters becomes counting length.
            continue

        first,second = counts.most_common(n=2)
        if first[1] != second[1]:  # Unique max
            yield first[1], string


def dataset_argmaxCharacter(generator: Iterable[str]):
    for string in generator:
        counts = Counter(string.lower())  # Count in lowercase

        # Filter weird characters
        keys = list(counts.keys())
        for key in keys:
            if not key.isalpha():
                counts.pop(key)

        if len(counts) < 3:  # This is a nonsense string. It doesn't even contain 3 different unique letters...
            continue

        # Only if the maximum count is unique do we use this string, and only if that maximum is a lowercase letter.
        first,second = counts.most_common(n=2)
        if first[1] != second[1] and first[0].islower():
            yield first[0], string


def take(n: int, generator: Iterable):
    i = 0
    for thing in generator:
        if i >= n:
            break
        yield thing
        i += 1

    if i != n:  # Not the same as "no break <=> bad" because when the generator yields exactly n items, you still quit the loop without a break but in that case it's fine.
        print(f"Tried to take {n} examples but only {i} were generated.")


def addHoldoutPrefix(total_size: int, generator: Iterable[tuple]):
    SPLITS = ["tr", "va", "te"]
    for i,tup in tqdm(enumerate(take(total_size, generator)), total=total_size):
        if i < 0.8*total_size:
            yield (SPLITS[0],) + tup
        elif i < 0.9*total_size:
            yield (SPLITS[1],) + tup
        else:
            yield (SPLITS[2],) + tup


if __name__ == "__main__":
    out_path = PATH_DATA / "probing" / "visual" / "odd_character_out_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_unicodeInsert(generateSentences(loop=False))))

    out_path = PATH_DATA / "probing" / "visual" / "odd_character_out_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_unicodeInsert(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))

    ##############################################################################
    from generation.binning import OrderPreservingOverflow, BiggestKeySmallestBinFirst

    binner = OrderPreservingOverflow(margin=1.1)

    out_path = PATH_DATA / "probing" / "visual" / "count_character_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_countCharacter(generateSentences(loop=False), examples_per_sentence=3)), lastsep="|")
    # histogramOfTsv(out_path, column=1)
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 4 is best
    binner.binTsv(out_path, column=1, k_bins=4)
    # binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    out_path = PATH_DATA / "probing" / "visual" / "count_character_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_countCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))), examples_per_sentence=3)), lastsep="|")
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 2 or 4
    binner.binTsv(out_path, column=1, k_bins=4)
    # binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    out_path = PATH_DATA / "probing" / "visual" / "max_count_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(80_000, dataset_maxCharacter(generateSentences(loop=False))))
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 4 is best
    binner.binTsv(out_path, column=1, k_bins=4)
    # binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    out_path = PATH_DATA / "probing" / "visual" / "max_count_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_maxCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))
    # testBinAmounts(binner, out_path, column=1, max_k=6)  # 4 is best
    binner.binTsv(out_path, column=1, k_bins=4)
    # binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=4)

    ##############################################################################################################

    binner = BiggestKeySmallestBinFirst()

    out_path = PATH_DATA / "probing" / "visual" / "argmax_count_sentences.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(80_000, dataset_argmaxCharacter(generateSentences(loop=False))))
    # testBinAmounts(binner, out_path, column=1, max_k=7)  # 5 is nice IF you subsample the 'e' and 't' examples to 8k (making the dataset 40k instead).
    binner.binTsv(out_path, column=1, k_bins=5)
    # binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=5)
    # filtered_path = limitTsvValueCount(binned_path, column=1, max_frequency=8_000)
    # binner.binTsv(filtered_path, column=1, k_bins=5)

    out_path = PATH_DATA / "probing" / "visual" / "argmax_count_words.txt"
    if not out_path.exists():
        tuplesToFile(out_path, addHoldoutPrefix(100_000, dataset_argmaxCharacter(generateWords(
            map(len, map(str.split, generateSentences(loop=False)))))))
    # testBinAmounts(binner, out_path, column=1, max_k=7)  # 5 is nice but you need to cap every class to about 9900
    binner.binTsv(out_path, column=1, k_bins=5)
    # binned_path = binner.applyBinsToTsv(out_path, column=1, k_bins=5)
    # filtered_path = limitTsvValueCount(binned_path, column=1, max_frequency=9900)
    # binner.binTsv(filtered_path, column=1, k_bins=5)
