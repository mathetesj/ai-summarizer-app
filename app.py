import streamlit as st
import os
import subprocess
import glob
import time
import soundfile as sf
import numpy as np
from openai import OpenAI
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, AudioRecorderFactory

# --- 1. 핵심 기능 함수들 (기존 로직 재사용) ---

def process_audio_and_summarize(api_keys, audio_file_path):
    """오디오 파일을 받아 전체 요약 과정을 처리하는 메인 함수"""
    openai_key, google_key = api_keys
    summary = ""
    temp_folder = f"temp_chunks_{int(time.time())}"

    with st.spinner("오디오 파일을 처리 중입니다. 잠시만 기다려주세요..."):
        try:
            client = OpenAI(api_key=openai_key)
            genai.configure(api_key=google_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            chunk_files = split_audio_with_ffmpeg(audio_file_path, temp_folder)
            
            if chunk_files:
                progress_bar = st.progress(0, text="음성을 텍스트로 변환 중...")
                full_transcript = transcribe_audio_chunks(client, chunk_files, progress_bar)
                
                if full_transcript:
                    st.progress(1.0, text="텍스트 변환 완료! Gemini로 요약을 시작합니다...")
                    prompt = f"다음 회의록 텍스트를 아래 형식에 맞춰 Markdown 양식으로 멋지게 요약해 주세요.\n\n[회의록 텍스트]\n{full_transcript}\n\n[요약 형식]\n### 📌 핵심 요약\n\n### 📝 상세 내용\n- \n\n### 🚀 Action Items\n- "
                    response = model.generate_content(prompt)
                    summary = response.text

        except Exception as e:
            st.error(f"처리 중 오류 발생: {e}")
        finally:
            cleanup_temp_folder(temp_folder)
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
    
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

# --- 2. 실시간 녹음을 위한 클래스 (streamlit-webrtc) ---

class AudioRecorderFactory(AudioRecorderFactory):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        self.frames.append(frame.to_ndarray())
        return frame

    def get_audio_data(self):
        if not self.frames:
            return None
        # 모든 오디오 프레임을 하나로 합치고 WAV 파일로 저장
        audio_data = np.concatenate([f.flatten() for f in self.frames])
        samplerate = self.frames[0].sample_rate
        
        # 임시 WAV 파일로 저장
        temp_wav_path = "temp_recording.wav"
        sf.write(temp_wav_path, audio_data, samplerate)
        return temp_wav_path

# --- 3. Streamlit UI 구성 ---

st.set_page_config(page_title="AI 회의록 요약", page_icon="🎙️", layout="wide")
st.title("🎙️ AI 회의록 요약 웹앱")

with st.sidebar:
    st.header("API 키 설정")
    openai_api_key = st.text_input("OpenAI API 키", type="password", placeholder="sk-...")
    google_api_key = st.text_input("Google AI API 키", type="password", placeholder="AIza...")
    st.info("API 키는 새로고침하면 사라지며, 서버에 저장되지 않습니다.")

tab1, tab2 = st.tabs(["📁 파일 업로드", "🔴 실시간 녹음"])

with tab1:
    st.subheader("이미 녹음된 파일을 업로드하세요.")
    uploaded_file = st.file_uploader("음성 파일 (MP3, M4A, WAV...)", type=['mp3', 'm4a', 'wav', 'mp4'])

    if uploaded_file:
        if st.button("파일로 요약 시작하기", key="upload_button"):
            if not openai_api_key or not google_api_key:
                st.warning("사이드바에 API 키를 먼저 입력해주세요!")
            else:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                process_audio_and_summarize((openai_api_key, google_api_key), uploaded_file.name)

with tab2:
    st.subheader("웹앱에서 바로 녹음을 시작하세요.")
    
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        audio_recorder_factory=AudioRecorderFactory,
        media_stream_constraints={"video": False, "audio": True},
        in_recorder=True,
        out_recorder=False,
    )

    if not webrtc_ctx.state.playing:
        if webrtc_ctx.audio_recorder_factory:
            audio_file_path = webrtc_ctx.audio_recorder_factory.get_audio_data()
            if audio_file_path:
                st.audio(audio_file_path)
                if st.button("이 녹음으로 요약 시작하기", key="record_button"):
                    if not openai_api_key or not google_api_key:
                        st.warning("사이드바에 API 키를 먼저 입력해주세요!")
                    else:

                        process_audio_and_summarize((openai_api_key, google_api_key), audio_file_path)

