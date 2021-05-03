import os
from typing import Tuple, Optional, Union
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
TEST_LABELS = ['yes', 'no', 'up', 'down',
               'left', 'right', 'on', 'off', 'stop', 'go', 'silence']


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip()))
                       for line in fileobj]
    return output


def load_speechcommands_item(filepath: str, path: str, test=True) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    waveform, sample_rate = torchaudio.load(filepath)

    if test and label not in TEST_LABELS:
        label = 'unknown'

    return waveform, sample_rate, label, speaker_id, utterance_number


class SPEECHCOMMANDS(Dataset):
    """Create a Dataset for Speech Commands.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
            (default: ``"speech_commands_v0.02"``)
        subset (Optional[str]):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://arxiv.org/pdf/1804.03209.pdf>`_. (Default: ``None``)
    """

    def __init__(self,
                 root: Union[str, Path],
                 subset: Optional[str] = None,
                 ) -> None:

#         assert subset is None or subset in ["training", "validation", "testing"], (
#             "When `subset` not None, it must take a value from "
#             + "{'training', 'validation', 'testing'}."
#         )

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        self._path = os.path.join(root, URL)

        if subset == "validation":
            self._walker = _load_list(self._path, "../validation_list.txt")
            self.test = True
        elif subset == "testing":
            self._walker = _load_list(self._path, "../testing_list.txt")
            self.test = True
        elif subset == "training":
            self.test = False
            excludes = set(_load_list(
                self._path, "../validation_list.txt", "../testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [
                w for w in walker
                if HASH_DIVIDER in w
                and EXCEPT_FOLDER not in w
                and os.path.normpath(w) not in excludes
            ]
       
        else:
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [
                w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path, self.test)

    def __len__(self) -> int:
        return len(self._walker)
