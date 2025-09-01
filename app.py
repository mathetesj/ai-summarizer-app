import streamlit as st
import os
import subprocess
import glob
import time
from openai import OpenAI
import google.generativeai as genai
from st_audiorecorder import st_audiorecorder # 오디오 녹음기 컴포넌트

# --- 1. 핵심 기능 함수들 (재사용을 위해 함수로 묶음) ---

def process_audio_and_summarize(api_keys, audio_file_path):
    """오디오 파일을 받아 전체 요약 과정을 처리하는 메인 함수"""
    openai_key, google_key = api_keys
    summary = ""
    temp_folder = f"temp_chunks_{int(time.time())}" # 동시 실행을 위해 고유 폴더 생성

    with st.spinner("오디오 파일을 처리 중입니다. 잠시만 기다려주세요..."):
        try:
            # 1. OpenAI, Google AI 클라이언트 설정
            client = OpenAI(api_key=openai_key)
            genai.configure(api_key=google_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # 2. 오디오 파일 분할
            chunk_files = split_audio_with_ffmpeg(audio_file_path, temp_folder)
            
            if chunk_files:
                # 3. 텍스트 변환
                progress_bar = st.progress(0, text="음성을 텍스트로 변환 중...")
                full_transcript = transcribe_audio_chunks(client, chunk_files, progress_bar)
                
                if full_transcript:
                    # 4. Gemini로 요약
                    st.progress(1.0, text="텍스트 변환 완료! Gemini로 요약을 시작합니다...")
                    prompt = f"다음 회의록 텍스트를 아래 형식에 맞춰 Markdown 양식으로 멋지게 요약해 주세요.\n\n[회의록 텍스트]\n{full_transcript}\n\n[요약 형식]\n### 📌 핵심 요약\n\n### 📝 상세 내용\n- \n\n### 🚀 Action Items\n- "
                    response = model.generate_content(prompt)
                    summary = response.text

        except Exception as e:
            st.error(f"처리 중 오류 발생: {e}")
        finally:
            # 5. 임시 파일 정리
            cleanup_temp_folder(temp_folder)
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path) # 업로드되거나 녹음된 임시 파일 삭제
    
    # 6. 최종 결과 표시
    if summary:
        st.success("✅ 요약이 완료되었습니다!")
        st.markdown("---")
        st.markdown(summary)
        st.balloons()
    else:
        st.error("요약 생성에 실패했습니다. 다시 시도해주세요.")

def split_audio_with_ffmpeg(file_path, temp_folder, chunk_duration_sec=1500):
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)
    try:
        command = f'ffmpeg -i "{file_path}" -f segment -segment_time {chunk_duration_sec} -c copy "{os.path.join(temp_folder, "chunk_%03d.m4a")}"'
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return sorted(glob.glob(os.path.join(temp_folder, "chunk_*.m4a")))
    except Exception as e:
        st.error(f"오디오 파일 분할 중 오류: {e}")
        return None

def transcribe_audio_chunks(client, chunk_files, progress_bar):
    full_transcript = ""
    total_chunks = len(chunk_files)
    for i, chunk_file in enumerate(chunk_files):
        try:
            with open(chunk_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
            full_transcript += transcript + " "
            progress_bar.progress((i + 1) / total_chunks, text=f"텍스트 변환 중... ({i+1}/{total_chunks})")
        except Exception as e:
            st.error(f"'{os.path.basename(chunk_file)}' 변환 중 오류: {e}")
    return full_transcript

def cleanup_temp_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder): os.remove(os.path.join(folder, file))
        os.rmdir(folder)

# --- 3. Streamlit UI 구성 ---

st.set_page_config(page_title="AI 회의록 요약", page_icon="🎙️", layout="wide")
st.title("🎙️ AI 회의록 요약 웹앱")

# API 키 입력 사이드바
with st.sidebar:
    st.header("API 키 설정")
    openai_api_key = st.text_input("OpenAI API 키", type="password", placeholder="sk-...")
    google_api_key = st.text_input("Google AI API 키", type="password", placeholder="AIza...")
    st.info("API 키는 새로고침하면 사라지며, 서버에 저장되지 않습니다.")

# '파일 업로드'와 '실시간 녹음' 탭으로 UI 분리
tab1, tab2 = st.tabs(["📁 파일 업로드", "🔴 실시간 녹음"])

with tab1:
    st.subheader("이미 녹음된 파일을 업로드하세요.")
    uploaded_file = st.file_uploader("음성 파일 (MP3, M4A, WAV...)", type=['mp3', 'm4a', 'wav', 'mp4'])

    if uploaded_file:
        if st.button("파일로 요약 시작하기", key="upload_button"):
            if not openai_api_key or not google_api_key:
                st.warning("사이드바에 API 키를 먼저 입력해주세요!")
            else:
                # 업로드된 파일을 서버에 임시 저장
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # 메인 처리 함수 호출
                process_audio_and_summarize((openai_api_key, google_api_key), uploaded_file.name)

with tab2:
    st.subheader("웹앱에서 바로 녹음을 시작하세요.")
    # 오디오 녹음기 컴포넌트 표시
    audio_bytes = st_audiorecorder(
        start_prompt="🔴 녹음 시작",
        stop_prompt="⏹️ 녹음 중지",
        pause_prompt="",
        icon_size="2rem"
    )

    if audio_bytes:
        # 녹음이 완료되면 오디오 데이터가 반환됨
        st.audio(audio_bytes, format="audio/wav")
        if st.button("녹음 파일로 요약 시작하기", key="record_button"):
            if not openai_api_key or not google_api_key:
                st.warning("사이드바에 API 키를 먼저 입력해주세요!")
            else:
                # 녹음된 오디오 데이터를 서버에 임시 파일로 저장
                recording_path = "temp_recording.wav"
                with open(recording_path, "wb") as f:
                    f.write(audio_bytes)
                # 메인 처리 함수 호출
                process_audio_and_summarize((openai_api_key, google_api_key), recording_path)

