

<br/>
<h2 align="center">Cam++ speaker verification with onnx</h2>
<br/>


![python3.7](https://img.shields.io/badge/python-3.7-green.svg)
![python3.8](https://img.shields.io/badge/python-3.8-green.svg)
![python3.9](https://img.shields.io/badge/python-3.9-green.svg)
![python3.10](https://img.shields.io/badge/python-3.10-green.svg)



  speaker verification from [modelscope](https://modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/summary) and [funasr](https://github.com/alibaba-damo-academy/FunASR/).



<br/>
<h2 align="center">Key Features</h2>
<br/>

- **Auto register speaker**

  support waveform bytes or wave file input, auto register speaker with
  threshold. output speaker id.

- **Lightweight**

  Do not need to download the model, the model is loaded from the memory directly.
  and the onnx model size is only **28M**.
  Do not need pytorch, torchaudio, etc. dependencies.




## Installation

```bash
git clone https://github.com/lovemefan/campplus
cd campplus
python setup.py install
```

## Usage
Input audio file or waveform bytes, output speaker id

speaker id in range of 0,1,2,...,n

```python
from campplus.src.Campplus import Campplus
audio_file1 = 'test/a_cn_16k.wav'
audio_file2 = 'test/b_cn_16k.wav'

# The higher the threshold, 
# the higher the degree of discrimination,
# and the more speaker identities are returned
# threshold 越高区分度越高，返回的说话人身份越多
# 
model = Campplus(threshold=0.7)
index = model.recognize(audio_file1)
print(index)
index = model.recognize(audio_file1)
print(index)
index = model.recognize(audio_file2)
print(index)
index = model.recognize(audio_file2)
print(index)
```
Output: 
```
0
0
1
1
```

## Citation
```
@article{cam++,
  title={CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking},
  author={Hui Wang and Siqi Zheng and Yafeng Chen and Luyao Cheng and Qian Chen},
  journal={arXiv preprint arXiv:2303.00332},
}
```