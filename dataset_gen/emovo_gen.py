import os
import re
import subprocess
import sys
import time
import uuid

import glob
import numpy as np
import librosa
import wavio

def main():
    data_path = os.path.join(sys.argv[1], "emovo")
    fs_list = [16000, 44100]

    # Convert sampling rate
    for fs in fs_list:
        convert_fs(
            os.path.join(data_path, "unprocessed"),
            os.path.join(data_path, f"wav{fs // 1000}"),
            fs,
        )

    # Create npz files
    for fs in fs_list:
        src_path = os.path.join(data_path, f"wav{fs // 1000}")

        create_dataset(src_path, os.path.join(data_path, f"wav{fs // 1000}.npz"))


def convert_fs(src_path, dst_path, fs):
    print(f"* {src_path} -> {dst_path}")
    os.mkdir(dst_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, "**", "*.wav"))):
        dst_file = src_file.replace(os.path.dirname(src_file), dst_path)
        subprocess.call(
            f"ffmpeg -i {src_file} -ac 1 -ar {fs} -loglevel error -y {dst_file}",
            shell=True,
        )


def create_dataset(src_path, dst_path):
    print(f"* {src_path} -> {dst_path}")
    classes = {"rab": 0, "gio": 1, "neu": 2, "tri": 3}
    dataset = {}

    sounds = []
    labels = []

    for wav_file in sorted(glob.glob(os.path.join(src_path, "*.wav"))):
        sound = wavio.read(wav_file).data.T[0]
        start = sound.nonzero()[0].min()
        end = sound.nonzero()[0].max()
        sound = sound[start : end + 1]
        file_name = os.path.splitext(os.path.basename(wav_file))[0]
        file_re = r"([a-z]+)\d+\-.*"
        label = re.match(file_re, file_name, re.IGNORECASE).group(1)

        if label in classes:
            sounds.append(sound)
            labels.append(classes.get(label))

    dataset["sounds"] = sounds
    dataset["labels"] = labels

    np.savez(dst_path, **dataset)


if __name__ == "__main__":
    main()
