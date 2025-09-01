import streamlit as st
import os
import subprocess
import glob
from openai import OpenAI
import google.generativeai as genai

# --- 1. 기능 함수들 (기존 로직 재사용) ---
def split_audio_with_ffmpeg(file_path, temp_folder, chunk_duration_sec=1500):
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)
    try:
        # ✨ 개선된 부분: 시스템에서 ffmpeg을 직접 찾도록 경로를 지정하지 않음
        command = f'ffmpeg -i "{file_path}" -f segment -segment_time {chunk_duration_sec} -c copy "{os.path.join(temp_folder, "chunk_%03d.m4a")}"'
        # subprocess.run에 shell=True 옵션을 추가하여 문자열 명령어를 직접 실행
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return sorted(glob.glob(os.path.join(temp_folder, "chunk_*.m4a")))
    except Exception as e:
        st.error(f"오디오 파일 분할 중 오류 발생: {e}")
        return None

def transcribe_audio_chunks(client, chunk_files, progress_bar):
    full_transcript = ""
    total_chunks = len(chunk_files)
    for i, chunk_file in enumerate(chunk_files):
        try:
            with open(chunk_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
            full_transcript += transcript + " "
            progress_bar.progress((i + 1) / total_chunks, text=f"음성을 텍스트로 변환 중... ({i+1}/{total_chunks})")
        except Exception as e:
            st.error(f"'{os.path.basename(chunk_file)}' 변환 중 오류: {e}")
    return full_transcript

def cleanup_temp_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder): os.remove(os.path.join(folder, file))
        os.rmdir(folder)

# --- 2. Streamlit UI 구성 ---
st.set_page_config(page_title="AI 회의록 요약", page_icon="🎙️")
st.title("🎙️ AI 회의록 요약 웹앱")

# API 키 입력 (사이드바 사용)
with st.sidebar:
    st.header("API 키 설정")
    openai_api_key = st.text_input("OpenAI API 키", type="password", placeholder="sk-...")
    google_api_key = st.text_input("Google AI API 키", type="password", placeholder="AIza...")
    st.info("API 키는 새로고침하면 사라지며, 서버에 저장되지 않습니다.")

# 파일 업로드
uploaded_file = st.file_uploader("요약할 음성 파일을 업로드하세요 (MP3, M4A, WAV...)", type=['mp3', 'm4a', 'wav', 'mp4'])

if uploaded_file is not None:
    if st.button("요약 시작하기"):
        if not openai_api_key or not google_api_key:
            st.warning("사이드바에 API 키를 먼저 입력해주세요!")
        else:
            # --- 3. 자동화 로직 실행 ---
            temp_folder = "temp_chunks_streamlit"
            with st.spinner("오디오 파일을 처리 중입니다. 잠시만 기다려주세요..."):
                # 1. 업로드된 파일 저장
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. 기능 실행
                client = OpenAI(api_key=openai_api_key)
                chunk_files = split_audio_with_ffmpeg(uploaded_file.name, temp_folder)
                
                summary = ""
                if chunk_files:
                    progress_bar = st.progress(0, text="음성을 텍스트로 변환 중...")
                    full_transcript = transcribe_audio_chunks(client, chunk_files)
                    
                    if full_transcript:
                        st.progress(1.0, text="텍스트 변환 완료! 요약을 시작합니다...")
                        genai.configure(api_key=google_api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = f"다음 회의록 텍스트를 아래 형식에 맞춰 요약해 주세요.\n\n[회의록 텍스트]\n{full_transcript}\n\n[요약 형식]\n### 📌 핵심 요약\n\n### 📝 상세 내용\n- \n\n### 🚀 Action Items\n- "
                        response = model.generate_content(prompt)
                        summary = response.text
                    
                    # 3. 임시 파일 정리
                    cleanup_temp_folder(temp_folder)
                    os.remove(uploaded_file.name)
            
            if summary:
                st.success("✅ 요약이 완료되었습니다!")
                st.markdown("---")
                st.markdown(summary)
                st.balloons() # 완료 축하!
            else:
                st.error("요약 생성에 실패했습니다. 파일을 다시 확인해주세요.")
