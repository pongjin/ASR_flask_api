# Flask API로 만든 간단한 모델 예측API
#### Architecture  
```
├── flask_api/
   ├── app.py
   ├── data/
     └── aa.wav
```
#### 필요 python 환경
```
pip install soundfile  
pip install PySoundFile  
pip install Flask  
pip install transformers  
pip install wave  
pip install torchaudio  
pip install torch  
pip install pyctcdecode
```
#### Process
aa.wav 입력 -> 10초 단위 분할 -> 변환 -> 모델 입력 -> 인식 결과 출력

#### How to use
```
# cmd
cd 개인경로/flask_api
set FLASK_APP=app.py
flask run             # Running on http://127.0.0.1:5000 뜨면 성공

# 다른 cmd창
cd 개인경로/flask_api/data/
curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@aa.wav"
# 뒤 aa.wav는 원하는 wav파일로 변경 가능

## 실행 결과
{"result":"fact to be named in march it may be the most important appointment governor micho d cachus makes during the remainder of his administration and one of the touhist as the"}
```
**현재 16khz로 저장된 .wav 파일만 가능(추후 수정 예정)** 
