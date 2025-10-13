import gradio as gr
from .main import TTSService
from .config import TTS_MODELS

def tts_demo(text, model_name, ref_audio, ref_text):
    service = TTSService(model_name=model_name)
    return service.generate_speech(text, ref_audio, ref_text)


with gr.Blocks() as demo:
    gr.Markdown("## Text-to-Speech Demo (Multi Provider/Model)")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text", placeholder="Enter text here...", lines=4)
            ref_audio_input = gr.Audio(label="Reference Audio (optional)", type="filepath")
            ref_text_input = gr.Textbox(label="Reference Text (optional)", placeholder="Optional reference text", lines=2)
            model_dropdown = gr.Dropdown(TTS_MODELS, value=TTS_MODELS[0], label="Provider / Model")
            generate_btn = gr.Button("Generate Speech")

        with gr.Column():
            output_audio = gr.Audio(label="Generated Audio")
            status = gr.Markdown()

    generate_btn.click(
        tts_demo,
        inputs=[text_input, model_dropdown, ref_audio_input, ref_text_input],
        outputs=[output_audio, status]
    )


def launch_ui(server_name="0.0.0.0", server_port=7860, share=False):
    demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch_ui()
