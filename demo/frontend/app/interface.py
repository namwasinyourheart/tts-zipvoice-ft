import gradio as gr
from .main import TTSService
from .config import TTS_MODELS

# Danh sách các file giọng tham chiếu có sẵn
EXAMPLE_VOICES = {
    "Lại Văn Sâm": "/home/nampv1/projects/tts/tts-ft/vnpost_tts_ft/examples/example_celeb_voices/laivansam_15s.mp3",
    "Mỹ Tâm": "/home/nampv1/projects/tts/tts-ft/vnpost_tts_ft/examples/example_celeb_voices/mytam_7s.mp3",
    "Sơn Tùng MTP": "/home/nampv1/projects/tts/tts-ft/vnpost_tts_ft/examples/example_celeb_voices/sontungmtp_2_10s.mp3",
    "Đen Vâu": "/home/nampv1/projects/tts/tts-ft/vnpost_tts_ft/examples/example_celeb_voices/denvau.mp3",
}


def tts_demo(text, model_name, selected_voice, uploaded_audio, ref_text):
    service = TTSService(model_name=model_name)
    # Ưu tiên audio người dùng upload, nếu không thì dùng audio có sẵn
    ref_audio = uploaded_audio or EXAMPLE_VOICES.get(selected_voice)

    audio_path, message = service.generate_speech(text, ref_audio, ref_text)
    if audio_path:
        return audio_path
    return None


def update_preview(selected_voice):
    """Cập nhật audio preview khi chọn giọng predefined"""
    return EXAMPLE_VOICES.get(selected_voice)


with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Text-to-Speech Demo")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text", placeholder="Enter text here...", lines=4
            )

            model_dropdown = gr.Dropdown(
                TTS_MODELS, value=TTS_MODELS[0], label="Model"
            )


            default_voice = list(EXAMPLE_VOICES.keys())[0]

            selected_voice = gr.Dropdown(
                choices=list(EXAMPLE_VOICES.keys()),
                value=list(EXAMPLE_VOICES.keys())[0],
                label="Select voice",
            )

            # Thêm trình phát preview giọng đã chọn
            voice_preview = gr.Audio(
                value=EXAMPLE_VOICES[default_voice],
                label="Preview selected voice", interactive=False
            )

            # Tự động cập nhật audio preview khi chọn giọng
            selected_voice.change(
                fn=update_preview, inputs=selected_voice, outputs=voice_preview
            )
            
            with gr.Accordion("Or use your own reference audio", open=False):

                ref_audio_input = gr.Audio(
                    label="Upload or Record your own reference audio",
                    type="filepath",
                    interactive=True,
                )

                ref_text_input = gr.Textbox(
                    label="Transcription of your reference audio (optional)",
                    placeholder="Optional transcription of your reference audio. An accurate transcription will improve the quality of the generated speech. If you don't provide this, the transcription will be generated automatically. ",
                    lines=2,
                )

            generate_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="Generated Audio", interactive=False)

    generate_btn.click(
        tts_demo,
        inputs=[
            text_input,
            model_dropdown,
            selected_voice,
            ref_audio_input,
            ref_text_input,
        ],
        outputs=[output_audio],
    )


def launch_ui(server_name="0.0.0.0", server_port=7860, share=False):
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        allowed_paths=[
            "/home/nampv1/projects/tts/tts-ft/vnpost_tts_ft/examples/example_celeb_voices"
        ],
    )


if __name__ == "__main__":
    launch_ui()
