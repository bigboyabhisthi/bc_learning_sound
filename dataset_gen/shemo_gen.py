"""
 Dataset preparation code for ESC-50 and ESC-10 [Piczak, 2015].
 Usage: python esc_gen.py [path]
 FFmpeg should be installed.

"""

import sys
import os
import subprocess

import glob
import numpy as np
import wavio


def main():
    data_path = os.path.join(sys.argv[1], "shemo")
    os.mkdir(data_path)
    fs_list = [16000, 44100]  # EnvNet and EnvNet-v2, respectively

    for g in ["female", "male"]:
        subprocess.call(
            f"wget -O {os.path.join(data_path, g + '.zip')} https://github.com/pariajm/sharif-emotional-speech-database/blob/master/{g}.zip?raw=true",
            shell=True,
        )
        subprocess.call(
            "unzip -d {} {}".format(
                os.path.join(data_path, g), os.path.join(data_path, f"{g}.zip")
            ),
            shell=True,
        )
        os.remove(os.path.join(data_path, f"{g}.zip"))

    # Convert sampling rate
    for fs in fs_list:
        convert_fs(
            os.path.join(data_path),
            os.path.join(data_path, "wav{}".format(fs // 1000)),
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
    classes = {"A": 0, "H": 1, "S": 2, "N": 3, "W": 4}
    dataset = {}

    sounds = []
    labels = []

    for wav_file in sorted(glob.glob(os.path.join(src_path, "*.wav"))):
        sound = wavio.read(wav_file).data.T[0]
        start = sound.nonzero()[0].min()
        end = sound.nonzero()[0].max()
        sound = sound[start : end + 1]
        label = os.path.splitext(os.path.basename(wav_file))[0][-3]

        if label in classes:
            sounds.append(sound)
            labels.append(classes.get(label))

    dataset["sounds"] = sounds
    dataset["labels"] = labels

    np.savez(dst_path, **dataset)


if __name__ == "__main__":
    main()
