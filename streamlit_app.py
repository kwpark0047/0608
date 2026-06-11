import os
import json
import base64
import tempfile
from datetime import datetime
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI 멀티모달 챗봇", page_icon="🎙️", layout="wide")
st.title("🎙️ AI 멀티모달 챗봇")
st.write("텍스트 · 음성 · 이미지로 GPT와 대화하세요.")

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    st.divider()
    st.subheader("🗣️ 음성")
    tts_enabled  = st.toggle("음성 응답 활성화", value=True)
    voice_option = st.selectbox(
        "TTS 음성",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        help="alloy=중성 / echo=남성 / fable=영국남성 / onyx=저음 / nova=여성 / shimmer=부드러운여성",
    )
    tts_model    = st.selectbox("TTS 모델", ["tts-1", "tts-1-hd"], help="tts-1-hd는 고품질·느림")
    stt_language = st.selectbox(
        "STT 언어", ["ko", "en", "ja", "zh"],
        format_func=lambda x: {"ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어"}[x],
    )

    st.divider()
    st.subheader("🖼️ 이미지")
    image_detail = st.selectbox("분석 정밀도", ["auto", "low", "high"],
                                help="high=고정밀(토큰 많음) / low=빠름 / auto=자동")

    st.divider()
    st.subheader("🤖 GPT")
    gpt_model     = st.selectbox("모델", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value="당신은 친절한 AI 어시스턴트입니다. 한국어로 답변해주세요.",
        height=100,
    )

    st.divider()
    st.subheader("💾 대화 저장/불러오기")

    def messages_to_json(messages):
        serializable = []
        for m in messages:
            entry = {"role": m["role"], "content": m["content"]}
            if "audio" in m:
                entry["audio"] = base64.b64encode(m["audio"]).decode()
            if "image_b64" in m:
                entry["image_b64"] = m["image_b64"]
                entry["image_mime"] = m.get("image_mime", "image/png")
            serializable.append(entry)
        return json.dumps(
            {"saved_at": datetime.now().isoformat(), "messages": serializable},
            ensure_ascii=False, indent=2,
        )

    def messages_to_txt(messages):
        lines = [f"[저장일시] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        for m in messages:
            role = "나" if m["role"] == "user" else "AI"
            img_note = " [이미지 첨부]" if "image_b64" in m else ""
            lines.append(f"[{role}]{img_note} {m['content']}\n")
        return "\n".join(lines)

    if st.session_state.get("messages"):
        fname = datetime.now().strftime("chat_%Y%m%d_%H%M%S")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("JSON 저장", messages_to_json(st.session_state.messages),
                               f"{fname}.json", "application/json", use_container_width=True)
        with c2:
            st.download_button("TXT 저장", messages_to_txt(st.session_state.messages),
                               f"{fname}.txt", "text/plain", use_container_width=True)
    else:
        st.caption("저장할 대화가 없습니다.")

    json_upload = st.file_uploader("JSON 불러오기", type=["json"], label_visibility="collapsed")
    if json_upload:
        try:
            data   = json.load(json_upload)
            loaded = []
            for m in data.get("messages", []):
                entry = {"role": m["role"], "content": m["content"]}
                if "audio"     in m: entry["audio"]      = base64.b64decode(m["audio"])
                if "image_b64" in m:
                    entry["image_b64"]  = m["image_b64"]
                    entry["image_mime"] = m.get("image_mime", "image/png")
                loaded.append(entry)
            st.session_state.messages = loaded
            st.success(f"{len(loaded)}개 메시지를 불러왔습니다.")
            st.rerun()
        except Exception as e:
            st.error(f"불러오기 실패: {e}")

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.pending_image   = None
        st.session_state.pending_mime    = None
        st.rerun()

# ── 사용자 정보 폼 ─────────────────────────────────────────
with st.expander("사용자 정보 입력", expanded=False):
    with st.form("user_form"):
        c1, c2 = st.columns(2)
        name  = c1.text_input("이름")
        topic = c2.text_input("질문 주제")
        if st.form_submit_button("제출", use_container_width=True):
            if name and topic:
                st.success(f"{name}님, '{topic}'에 대한 질문을 받았습니다!")
            else:
                st.warning("이름과 질문 주제를 모두 입력해주세요.")

st.divider()

if not openai_api_key:
    st.info("왼쪽 사이드바에 OpenAI API 키를 입력하면 챗봇을 사용할 수 있습니다.", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

if "messages"      not in st.session_state: st.session_state.messages      = []
if "pending_image" not in st.session_state: st.session_state.pending_image = None
if "pending_mime"  not in st.session_state: st.session_state.pending_mime  = None

# ── 헬퍼 함수 ─────────────────────────────────────────────
def encode_image(file_bytes: bytes, mime: str) -> str:
    return base64.b64encode(file_bytes).decode()

def build_user_content(text: str, image_b64: str | None, mime: str | None, detail: str):
    if not image_b64:
        return text
    return [
        {"type": "text", "text": text or "이 이미지를 분석해주세요."},
        {"type": "image_url", "image_url": {
            "url":    f"data:{mime};base64,{image_b64}",
            "detail": detail,
        }},
    ]

def generate_reply(client, messages, model, system_prompt, image_b64=None, mime=None, detail="auto", stream=True):
    history = [{"role": "system", "content": system_prompt}]
    for m in messages[:-1]:
        history.append({"role": m["role"], "content": m["content"]})
    last = messages[-1]
    history.append({
        "role": "user",
        "content": build_user_content(last["content"], image_b64, mime, detail),
    })
    vision_models = {"gpt-4o", "gpt-4o-mini"}
    if image_b64 and model not in vision_models:
        model = "gpt-4o"
    return client.chat.completions.create(model=model, messages=history, stream=stream)

def generate_tts(client, text, model, voice):
    return client.audio.speech.create(model=model, voice=voice, input=text).content

def transcribe(client, audio_bytes, language):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-1", file=f, language=language)
        return result.text
    finally:
        os.unlink(tmp_path)

# ── 대화 히스토리 출력 ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image_b64" in msg:
            img_bytes = base64.b64decode(msg["image_b64"])
            st.image(img_bytes, caption="첨부 이미지", use_container_width=True)
        st.markdown(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# ── 이미지 입력 영역 ──────────────────────────────────────
st.subheader("🖼️ 이미지 입력")
img_tab1, img_tab2 = st.tabs(["파일 업로드", "카메라 촬영"])

with img_tab1:
    uploaded_img = st.file_uploader(
        "이미지를 업로드하세요 (JPG · PNG · WEBP · GIF)",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        label_visibility="collapsed",
    )
    if uploaded_img:
        st.session_state.pending_image = uploaded_img.read()
        st.session_state.pending_mime  = uploaded_img.type
        st.image(st.session_state.pending_image, caption="업로드된 이미지", use_container_width=True)

with img_tab2:
    camera_img = st.camera_input("카메라로 촬영하세요")
    if camera_img:
        st.session_state.pending_image = camera_img.read()
        st.session_state.pending_mime  = "image/jpeg"

if st.session_state.pending_image:
    col_info, col_clear = st.columns([4, 1])
    col_info.success("이미지가 준비됐습니다. 아래에서 질문하면 함께 전송됩니다.")
    if col_clear.button("이미지 제거"):
        st.session_state.pending_image = None
        st.session_state.pending_mime  = None
        st.rerun()

st.divider()

# ── 입력 방식 선택 ────────────────────────────────────────
input_mode = st.radio("입력 방식", ["텍스트", "음성"], horizontal=True)

# ── 공통: 메시지 전송 처리 ────────────────────────────────
def handle_send(prompt: str):
    image_b64 = encode_image(st.session_state.pending_image, st.session_state.pending_mime) \
                if st.session_state.pending_image else None
    mime      = st.session_state.pending_mime

    user_msg = {"role": "user", "content": prompt}
    if image_b64:
        user_msg["image_b64"]  = image_b64
        user_msg["image_mime"] = mime
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        if image_b64:
            st.image(base64.b64decode(image_b64), caption="첨부 이미지", use_container_width=True)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("응답 생성 중..."):
            stream = generate_reply(
                client, st.session_state.messages, gpt_model,
                system_prompt, image_b64, mime, image_detail,
            )
        reply = st.write_stream(stream)

    # 이미지 전송 후 초기화
    st.session_state.pending_image = None
    st.session_state.pending_mime  = None

    asst_msg = {"role": "assistant", "content": reply}
    if tts_enabled:
        with st.spinner("음성 응답 생성 중..."):
            audio_bytes = generate_tts(client, reply, tts_model, voice_option)
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        asst_msg["audio"] = audio_bytes
    st.session_state.messages.append(asst_msg)

# ── 음성 입력 ─────────────────────────────────────────────
if input_mode == "음성":
    audio_value = st.audio_input("마이크 버튼을 눌러 말씀하세요")
    if audio_value:
        with st.spinner("음성 인식 중..."):
            prompt = transcribe(client, audio_value.getvalue(), stt_language)
        if prompt:
            st.info(f"🗣️ 인식된 텍스트: **{prompt}**")
            handle_send(prompt)

# ── 텍스트 입력 ───────────────────────────────────────────
else:
    if prompt := st.chat_input("메시지를 입력하세요..."):
        handle_send(prompt)
