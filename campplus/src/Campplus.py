from typing import Union

import onnxruntime
import numpy as np
from pathlib import Path

from campplus.src.utils.AudioProcess import extract_feature


class Campplus:
    def __init__(self, onnx_path=None, threshold=0.5):
        """
        :param onnx_path: onnx model file path
        :param threshold: threshold of speaker embedding similarity
        """
        self.onnx = onnx_path or Path(__file__).parent.parent / 'onnx/campplus.onnx'
        self.sess = onnxruntime.InferenceSession(self.onnx)
        self.output_name = [nd.name for nd in self.sess.get_outputs()]
        self.threshhold = threshold
        self.memory: np.ndarray = None

    def compute_cos_similarity(self, emb):

        assert len(emb.shape) == 2, "emb must be length * 80"
        cos_sim = emb.dot(self.memory.T) / np.linalg.norm(emb)
        return cos_sim

    def register_speaker(self, emb: np.ndarray):
        """
        register speaker with embedding and name
        :param emb:
        :param name: speaker name
        :return:
        """
        assert len(emb.shape) == 2, "emb must be length * 80"
        self.memory = np.concatenate((self.memory, emb / np.linalg.norm(emb), ))

    def embedding(self, feature: np.ndarray):
        feed_dict = {
            'fbank': feature
        }
        output = self.sess.run(self.output_name, input_feed=feed_dict)
        return output

    def recognize(self, waveform: Union[str, Path, bytes]):
        """
        auto register speaker with input waveform。
        input waveform, output speaker id , id in range 0,1,2,....,n
        :param waveform:
        :return index: if max similarity less than threshold, it will add current emb into memory
        """
        feature = extract_feature(waveform)
        emb = self.embedding(feature)[0]

        if self.memory is None:
            self.memory = emb / np.linalg.norm(emb)
            return 0
        sim = self.compute_cos_similarity(emb)[0]
        max_sim_index = np.argmax(sim)

        if sim[max_sim_index] <= self.threshhold:
            self.register_speaker(emb)

        return self.memory.shape[0] - 1


def onnx_text():

    audio_file1 = '/Users/cenglingfan/Downloads/a_cn_16k.wav'
    audio_file2 = '/Users/cenglingfan/Downloads/b_cn_16k.wav'
    audio_file3 = '/Users/cenglingfan/Downloads/wave_test.wav'

    # threshold 越高区分度越高，返回的说话人身份越多
    model = Campplus(threshold=0.5)
    index = model.recognize(audio_file1)
    print(index)
    index = model.recognize(audio_file1)
    print(index)
    index = model.recognize(audio_file2)
    print(index)
    index = model.recognize(audio_file2)
    print(index)
    index = model.recognize(audio_file3)
    print(index)
    index = model.recognize(audio_file3)
    print(index)

if __name__ == '__main__':
    
    onnx_text()