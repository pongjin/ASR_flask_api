import io
import json

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)

# 사전 학습된 모델과 토크나이저 경로
model_name = "pongjin/en_with_korean_w2v_model_960h"
tokenizer_name = "pongjin/en_with_korean_w2v_model_960h"

# 모델과 토크나이저 로드
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(tokenizer_name)

def slicing(waveform, sample_rate):

    # 잘라낼 시간 간격 설정 (초 단위)
    interval = 10

    # 잘라낸 오디오 데이터 저장할 리스트
    sliced_audio_data = []

    # 오디오 데이터를 interval 간격으로 자르기
    start_time = 0
    end_time = interval * sample_rate
    while end_time <= waveform.size(1):
        sliced_audio_data.append(waveform[:, start_time:end_time])
        start_time += interval * sample_rate
        end_time += interval * sample_rate
    return sliced_audio_data


def get_prediction(sliced_audio_data):
    total = []
    # 잘라진 오디오 데이터를 모델의 입력값으로 사용
    for sliced in sliced_audio_data:

        # 토큰화 및 인코딩
        input_values = tokenizer(sliced, return_tensors="pt").input_values
        input_values = torch.tensor(input_values[0])
        # 모델에 입력 텐서 전달하여 예측
        outputs = model(input_values)

        # CTC 디코딩을 통해 최종 예측 텍스트 얻기
        predicted_ids = outputs.logits.argmax(dim=-1)
        predicted_text = tokenizer.batch_decode(predicted_ids)[0]
        total.append(predicted_text)
    total = "".join(total)
    return total

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_file = request.files['file']
        # 앞서 16khz로 변경된 WAV 파일 로드
        waveform, sample_rate = torchaudio.load(input_file)
        sliced = slicing(waveform, sample_rate)
        transcript = get_prediction(sliced)
        
        return jsonify({'result': transcript})


if __name__ == '__main__':
    app.run()