# -*- coding:utf-8 -*-
# @FileName  :AudioProcess.py
# @Time      :2023/6/26 09:33
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import struct
from pathlib import Path
from typing import Union

import numpy as np
import array
import kaldi_native_fbank as knf


def read_wav_bytes(data: bytes):
    """
    convert bytes into array of pcm_s16le data
    :param data: PCM format bytes
    :return:
    """

    # header of wav file
    info = data[:44]
    frames = data[44:]
    (
        name,
        data_lengths,
        _,
        _,
        _,
        _,
        channels,
        sample_rate,
        bit_rate,
        block_length,
        sample_bit,
        _,
        pcm_length,
    ) = struct.unpack_from("<4sL4s4sLHHLLHH4sL", info)
    # shortArray each element is 16bit
    short_array = array.array("h")
    short_array.frombytes(data)
    data = np.array(short_array, dtype="float16") / (1 << 15)
    return data, sample_rate


def read_wav_file(audio_path: str):
    with open(audio_path, "rb") as f:
        data = f.read()
    return read_wav_bytes(data)


def extract_feature(audio: Union[str, Path, bytes]):
    if isinstance(audio, str) or isinstance(audio, Path):
        waveform, sample_rate = read_wav_file(audio)
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_rate)
    opts.frame_opts.dither = 0.0
    opts.energy_floor = 1.0
    opts.mel_opts.num_bins = 80
    fbank_fn = knf.OnlineFbank(opts)
    fbank_fn.accept_waveform(sample_rate, waveform.tolist())
    frames = fbank_fn.num_frames_ready
    mat = np.empty([frames, opts.mel_opts.num_bins])
    for i in range(frames):
        mat[i, :] = fbank_fn.get_frame(i)
    feature = mat.astype(np.float32)

    feature = feature - feature.mean()
    feature = feature[None, ...]
    return feature