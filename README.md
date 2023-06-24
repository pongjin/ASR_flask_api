# Flask로 만든 간단한 음성인식 모델 API
[에듀테크](https://github.com/EduTechProjects) 모델 API 부분
#### Architecture  
```
├── flask_api/
   ├── app.py
   └── audiofile2.wav
```
app.py : 직접 학습한 wav2vec + LM 모델을 허깅페이스에서 불러와 음성 인식 후 GPT api를 사용하여 문맥과 문법을 교정  
https://huggingface.co/pongjin/en_with_korean_model_large_960h  
audiofile2.wav : 예시 파일
#### 필요 python 환경

**kenlm 빌드로 인해 로컬에서는 현재 Linux랑 mac만 사용가능** 
```
pip install openai
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install soundfile  
pip install PySoundFile  
pip install Flask  
pip install transformers  
pip install wave  
pip install torchaudio  
pip install torch  
pip install pyctcdecode
pip install huggingface_hub
pip install re
```
#### Process
.wav 입력 -> 16Khz 변환 -> 10초 단위 분할 -> 변환 -> 모델 입력 -> 인식 결과 출력 -> GPT3.5 turbo로 문맥, 문법 수정 -> 인식결과, 수정결과 반환

#### How to use
```
# cmd
cd 개인경로/flask_api
set FLASK_APP=app.py
flask run             # Running on http://127.0.0.1:5000 뜨면 성공

# 다른 cmd창
cd 개인경로/flask_api/
curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/transcribe -F "file=@audiofile2.wav"
# 뒤 aa.wav는 원하는 wav파일로 변경 가능

## 실행 결과
{"result":"fact to be named in march it may be the most important appointment governor micho d cachus makes during the remainder of his administration and one of the touhist as the",
"fix": abc ~~"}
```
<img src='https://velog.velcdn.com/images/pong_jin/post/db810e83-6e3d-4097-9739-89d703d7da44/image.png'>
