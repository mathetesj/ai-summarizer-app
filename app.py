import streamlit as st
import os
import subprocess
import glob
from openai import OpenAI
import google.generativeai as genai

# --- 1. 기능 함수들 (핵심 로직) ---

def split_audio_with_ffmpeg(file_path, temp_folder, chunk_duration_sec=1500):
    """ffmpeg을 직접 호출하여 오디오 파일을 작은 조각으로 자르는 함수"""
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    try:
        # Streamlit 환경에서 ffmpeg을 안정적으로 호출하기 위한 명령어 형식
        command = f'ffmpeg -i "{file_path}" -f segment -segment_time {chunk_duration_sec} -c copy "{os.path.join(temp_folder, "chunk_%03d.m4a")}"'
        
        # shell=True 옵션을 사용하여 명령어를 직접 실행
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return sorted(glob.glob(os.path.join(temp_folder, "chunk_*.m4a")))
    except Exception as e:
        st.error(f"오디오 파일 분할 중 오류가 발생했습니다: {e}")
        return None

def transcribe_audio_chunks(client, chunk_files, progress_bar):
    """나눠진 오디오 조각들을 텍스트로 변환하고 진행 상황을 표시하는 함수"""
    full_transcript = ""
    total_chunks = len(chunk_files)
    for i, chunk_file in enumerate(chunk_files):
        try:
            with open(chunk_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
            full_transcript += transcript + " "
            # 진행 상황 업데이트
            progress_bar.progress((i + 1) / total_chunks, text=f"음성을 텍스트로 변환 중... ({i+1}/{total_chunks})")
        except Exception as e:
            st.error(f"'{os.path.basename(chunk_file)}' 변환 중 오류가 발생했습니다: {e}")
    return full_transcript

def cleanup_temp_folder(folder):
    """임시 조각 파일들과 폴더를 삭제하는 함수"""
    if os.path.exists(folder):
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)

# --- 2. Streamlit 웹앱 UI 구성 ---

# 페이지 기본 설정
st.set_page_config(page_title="AI 회의록 요약", page_icon="🎙️")
st.title("🎙️ AI 회의록 요약 웹앱")

# API 키 입력을 위한 사이드바
with st.sidebar:
    st.header("API 키 설정")
    openai_api_key = st.text_input("OpenAI API 키", type="password", placeholder="sk-...")
    google_api_key = st.text_input("Google AI API 키", type="password", placeholder="AIza...")
    st.info("API 키는 새로고침하면 사라지며, 서버에 저장되지 않습니다.")

# 파일 업로드 컴포넌트
uploaded_file = st.file_uploader("요약할 음성 파일을 업로드하세요 (MP3, M4A, WAV...)", type=['mp3', 'm4a', 'wav', 'mp4'])

# '요약 시작하기' 버튼이 눌렸을 때의 로직
if uploaded_file is not None:
    if st.button("요약 시작하기"):
        if not openai_api_key or not google_api_key:
            st.warning("사이드바에 API 키를 먼저 입력해주세요!")
        else:
            temp_folder = "temp_chunks_streamlit"
            with st.spinner("오디오 파일을 처리 중입니다. 잠시만 기다려주세요..."):
                
                # 1. 업로드된 파일을 임시로 서버에 저장
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. 자동화 기능 실행
                client = OpenAI(api_key=openai_api_key)
                chunk_files = split_audio_with_ffmpeg(uploaded_file.name, temp_folder)
                
                summary = ""
                if chunk_files:
                    progress_bar = st.progress(0, text="음성을 텍스트로 변환 중...")
                    full_transcript = transcribe_audio_chunks(client, chunk_files, progress_bar)
                    
                    if full_transcript:
                        st.progress(1.0, text="텍스트 변환 완료! Gemini로 요약을 시작합니다...")
                        genai.configure(api_key=google_api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = f"다음 회의록 텍스트를 아래 형식에 맞춰 Markdown 양식으로 멋지게 요약해 주세요.\n\n[회의록 텍스트]\n{full_transcript}\n\n[요약 형식]\n### 📌 핵심 요약\n\n### 📝 상세 내용\n- \n\n### 🚀 Action Items\n- "
                        response = model.generate_content(prompt)
                        summary = response.text
                    
                    # 3. 모든 작업이 끝난 후 임시 파일 정리
                    cleanup_temp_folder(temp_folder)
                    os.remove(uploaded_file.name)
            
            # 4. 최종 결과 표시
            if summary:
                st.success("✅ 요약이 완료되었습니다!")
                st.markdown("---")
                st.markdown(summary) # Markdown 형식으로 요약 내용을 예쁘게 표시
                st.balloons() # 완료 축하 효과!
            else:
                st.error("요약 생성에 실패했습니다. 파일을 다시 확인해주세요.")
