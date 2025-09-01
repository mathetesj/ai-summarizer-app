    # 1. 베이스 이미지 선택 (파이썬 3.9 버전)
    FROM python:3.9-slim
    
    # 2. 작업 폴더 설정
    WORKDIR /app
    
    # 3. 외부 프로그램 설치 (가장 중요! 여기서 ffmpeg을 설치합니다)
    RUN apt-get update && apt-get install -y ffmpeg
    
    # 4. 현재 폴더의 모든 파일을 작업 폴더로 복사
    COPY . .
    
    # 5. requirements.txt에 있는 파이썬 라이브러리 설치
    RUN pip install -r requirements.txt
    
    # 6. 앱 실행 명령어 (Hugging Face에서 사용하는 포트 7860으로 실행)
    CMD ["streamlit", "run", "app.py", "--server.port=7860"]
  
