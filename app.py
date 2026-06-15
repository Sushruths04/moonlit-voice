"""DreamVoice — Gradio 6 entry-point for the bedtime-story voice-cloning app.

HuggingFace ZeroGPU compatible: all GPU functions decorated with @spaces.GPU.
Models are loaded lazily at module level (outside @spaces.GPU) for CUDA emulation.
"""

from __future__ import annotations

import os

import spaces
import gradio as gr

from audio_utils import prepare_reference
from story import generate_story, VALID_GENRES, VALID_MOODS
from tts import clone_and_speak
from indic_text import translate_to_kannada
from indic_tts import narrate_kannada
from ui_html import starfield_background, storybook_html, storybook_markup

GENRE_CHOICES = [
    ("🦁 Animals", "animals"), ("👑 Kingdom", "kingdom"), ("🚀 Space", "space"),
    ("🐉 Dragons", "dragons"), ("🌊 Ocean", "ocean"), ("🌲 Forest", "forest"),
]
MOOD_CHOICES = [
    ("✨ Magical", "magical"), ("😄 Funny", "funny"), ("😴 Calming", "calming"), ("🌙 Dreamy", "dreamy"),
]
LANGUAGE_MAP = {"English": "en", "ಕನ್ನಡ Kannada": "kn"}

REF_SENTENCE = "Once upon a time, in a cozy little house, a gentle voice began to tell a bedtime story."


def _stage_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=Pacifico&display=swap');

.dv-section-title {
  font-family: 'Pacifico', cursive; font-size: 1.5rem; color: #f5c842;
  text-align: center; margin: .5rem 0 .8rem;
}
.dv-note {
  font-family: 'Lora', Georgia, serif; font-size: .85rem;
  color: rgba(255,255,255,.55); text-align: center; margin-top: .2rem;
}

.dv-radio-group fieldset { border: none !important; padding: 0 !important; }
.dv-radio-group legend { display: none !important; }
.dv-radio-group label {
  display: inline-block !important; margin: 4px !important; padding: 8px 18px !important;
  border-radius: 24px !important; border: 1.5px solid rgba(255,255,255,.15) !important;
  background: rgba(255,255,255,.06) !important; color: #ccc !important;
  font-family: 'Lora', Georgia, serif !important; font-size: .9rem !important;
  cursor: pointer; transition: all .15s !important;
}
.dv-radio-group label:hover {
  background: rgba(245,200,66,.12) !important; border-color: rgba(245,200,66,.4) !important;
  color: #eee !important;
}
.dv-radio-group input:checked + span { color: #f5c842 !important; font-weight: 700; }
.dv-radio-group label:has(input:checked) {
  background: rgba(245,200,66,.18) !important; border-color: #f5c842 !important; color: #f5c842 !important;
}

.dv-tiles label {
  font-size: 1rem !important; padding: 14px 20px !important; margin: 6px !important;
}
.dv-tiles label:has(input:checked) {
  transform: scale(1.06); box-shadow: 0 0 18px rgba(245,200,66,.45) !important;
}

.dv-btn-accent {
  background: #f5c842 !important; color: #1a1a3e !important; border: none !important;
  border-radius: 24px !important; padding: .65rem 1.8rem !important; font-weight: 700 !important;
  font-size: 1rem !important; font-family: 'Lora', Georgia, serif !important;
  transition: transform .15s, box-shadow .15s !important;
}
.dv-btn-accent:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(245,200,66,.4) !important; }

.dv-readbox {
  max-width: 620px; margin: .4rem auto 1rem; padding: 1rem 1.3rem;
  border-radius: 18px; text-align: center;
  background: rgba(245,200,66,.08); border: 1.5px dashed rgba(245,200,66,.35);
}
.dv-readbox-tag {
  display: block; font-family: 'Lora', Georgia, serif; font-size: .72rem;
  letter-spacing: .08em; text-transform: uppercase; color: rgba(245,200,66,.75);
  margin-bottom: .45rem;
}
.dv-readbox-text {
  display: block; font-family: 'Lora', Georgia, serif; font-style: italic;
  font-size: 1.05rem; line-height: 1.55; color: #f3efe0;
}

.dv-player { border-radius: 18px !important; box-shadow: 0 0 24px rgba(245,200,66,.18) !important; }

.dv-slider { max-width: 460px; margin: 0 auto .4rem; }

.dv-error {
  max-width: 560px; margin: .2rem auto 0; padding: .7rem 1rem; border-radius: 14px;
  text-align: center; font-family: 'Lora', Georgia, serif; color: #ffd9d0;
  background: rgba(255,120,90,.12); border: 1px solid rgba(255,120,90,.35);
}

.dv-loading { text-align: center; padding: 3rem 1rem 2rem; }
.dv-loading-moon { font-size: 4rem; animation: dv-bob 3s ease-in-out infinite; }
@keyframes dv-bob { 0%,100% { transform: translateY(0) rotate(-6deg); } 50% { transform: translateY(-12px) rotate(6deg); } }
.dv-zzz { font-family: 'Pacifico', cursive; color: #f5c842; font-size: 1.6rem; height: 1.8rem; }
.dv-zzz span { display: inline-block; opacity: 0; animation: dv-zzz 2.4s ease-in-out infinite; }
.dv-zzz span:nth-child(2) { animation-delay: .4s; font-size: 1.9rem; }
.dv-zzz span:nth-child(3) { animation-delay: .8s; font-size: 2.2rem; }
@keyframes dv-zzz { 0% { opacity: 0; transform: translateY(8px); } 40% { opacity: 1; } 100% { opacity: 0; transform: translateY(-10px); } }
.dv-loading-text { font-family: 'Pacifico', cursive; color: #f3efe0; font-size: 1.3rem; margin-top: .6rem; }

.dv-wave { display: flex; gap: 5px; justify-content: center; align-items: flex-end; height: 42px; margin: .8rem 0 .2rem; }
.dv-wave span {
  width: 5px; border-radius: 3px; background: linear-gradient(#f5c842, #ffaa00);
  animation: dv-wave 1.1s ease-in-out infinite;
}
.dv-wave span:nth-child(odd)  { animation-duration: .9s; }
.dv-wave span:nth-child(3n)   { animation-duration: 1.4s; }
.dv-wave span:nth-child(1){height:14px} .dv-wave span:nth-child(2){height:28px}
.dv-wave span:nth-child(3){height:40px} .dv-wave span:nth-child(4){height:24px}
.dv-wave span:nth-child(5){height:34px} .dv-wave span:nth-child(6){height:18px}
.dv-wave span:nth-child(7){height:40px} .dv-wave span:nth-child(8){height:26px}
.dv-wave span:nth-child(9){height:16px}
@keyframes dv-wave { 0%,100% { transform: scaleY(.4); opacity: .7; } 50% { transform: scaleY(1); opacity: 1; } }

.dv-footer {
  text-align: center; padding: 1.5rem 0 .5rem; font-family: 'Lora', Georgia, serif;
  font-size: .82rem; color: rgba(255,255,255,.35);
}
.dv-footer a {
  color: rgba(245,200,66,.55); text-decoration: none; transition: color .15s;
}
.dv-footer a:hover { color: #f5c842; }
</style>
"""


@spaces.GPU(duration=120)
def _tell_story(audio_path, genre, mood, hero_name, language_label, length, energy):
    """Full pipeline: prepare ref -> story -> clone & speak. Runs on ZeroGPU."""
    errors = []
    if not audio_path:
        errors.append("Please record or upload a voice clip first.")
    if genre not in VALID_GENRES:
        errors.append("Please choose a genre.")
    if mood not in VALID_MOODS:
        errors.append("Please choose a mood.")
    if errors:
        raise ValueError(" ".join(errors))

    lang_code = LANGUAGE_MAP.get(language_label, "en")
    energy_f = max(0.0, min(1.0, float(energy) / 100.0))
    ref_wav = None
    try:
        ref_wav = prepare_reference(audio_path)
        english_story = generate_story(
            genre, mood, hero_name.strip() if hero_name else "",
            language="en", length=length, energy=energy_f,
        )

        if lang_code == "kn":
            kannada_story = translate_to_kannada(english_story)
            narration_path = narrate_kannada(ref_wav, REF_SENTENCE, kannada_story, mood=mood, energy=energy_f)
            display = storybook_markup(kannada_story, "ಇಂದಿನ ಕಥೆ · Tonight's Story")
        else:
            narration_path = clone_and_speak(ref_wav, english_story, mood=mood, energy=energy_f)
            display = storybook_markup(english_story, "Tonight's Story")

        return display, narration_path
    finally:
        if ref_wav and os.path.exists(ref_wav):
            try:
                os.unlink(ref_wav)
            except OSError:
                pass


def _run_story(audio_path, genre, mood, hero_name, language_label, length, energy):
    """UI generator: loading scene -> pipeline -> reveal/error."""
    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        gr.update(visible=False, value=""),
    )
    try:
        display, audio = _tell_story(audio_path, genre, mood, hero_name, language_label, length, energy)
    except Exception as exc:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(), gr.update(),
            gr.update(visible=True, value=f'<div class="dv-error">😴 {exc}</div>'),
        )
        return
    yield (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        display, audio,
        gr.update(visible=False, value=""),
    )


def _build_app() -> gr.Blocks:
    with gr.Blocks(title="DreamVoice — Bedtime Stories in Mom's Voice") as demo:
        starfield_background()

        with gr.Column(visible=True) as stage1:
            gr.HTML('<h2 class="dv-section-title">🎙️ Record Mom\'s Voice</h2>')
            gr.HTML(
                '<div class="dv-readbox">'
                '<span class="dv-readbox-tag">Read this aloud, slowly &amp; clearly:</span>'
                f'<span class="dv-readbox-text">&ldquo;{REF_SENTENCE}&rdquo;</span>'
                "</div>"
            )
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload a 5–60 s voice clip",
            )
            gr.HTML(
                '<p class="dv-note">'
                "Find a quiet room. Reading the sentence above lets us narrate Kannada in this "
                "voice. Your recording is processed for this session only and never saved."
                "</p>"
            )

            gr.HTML('<p class="dv-section-title" style="font-size:1.15rem;margin-top:1rem;">Choose a genre</p>')
            genre_radio = gr.Radio(
                choices=GENRE_CHOICES,
                value="animals",
                label="",
                elem_classes=["dv-radio-group", "dv-tiles"],
            )

            gr.HTML('<p class="dv-section-title" style="font-size:1.15rem;">Pick a mood</p>')
            mood_radio = gr.Radio(
                choices=MOOD_CHOICES,
                value="magical",
                label="",
                elem_classes=["dv-radio-group", "dv-tiles"],
            )

            hero_name_input = gr.Textbox(
                label="Hero's name (optional)",
                placeholder="e.g. Luna",
                max_lines=1,
            )

            gr.HTML('<p class="dv-section-title" style="font-size:1.15rem;">Story language</p>')
            language_radio = gr.Radio(
                choices=["English", "ಕನ್ನಡ Kannada"],
                value="English",
                label="",
                elem_classes=["dv-radio-group"],
            )

            gr.HTML('<p class="dv-section-title" style="font-size:1.15rem;">Story length</p>')
            length_radio = gr.Radio(
                choices=["1 min", "2 min", "3 min"],
                value="2 min",
                label="",
                elem_classes=["dv-radio-group"],
            )

            gr.HTML('<p class="dv-section-title" style="font-size:1.15rem;">Delivery — 😴 calm ↔ energetic 🎉</p>')
            energy_slider = gr.Slider(
                minimum=0, maximum=100, value=45, step=5,
                label="", show_label=False, elem_classes=["dv-slider"],
            )

            error_box = gr.HTML(visible=False)
            tell_btn = gr.Button("Tell Me a Story ✨", elem_classes=["dv-btn-accent"])

        with gr.Column(visible=False) as loading_panel:
            gr.HTML(
                '<div class="dv-loading">'
                '<div class="dv-loading-moon">🌙</div>'
                '<div class="dv-zzz"><span>z</span><span>z</span><span>z</span></div>'
                '<p class="dv-loading-text">Tucking you in… weaving tonight\'s story</p>'
                '<p class="dv-note">Writing the story, then narrating it in your voice — '
                "this can take a little while.</p>"
                "</div>"
            )

        with gr.Column(visible=False) as stage2:
            story_display = storybook_html("", "Tonight's Story")
            audio_player = gr.Audio(
                label="🌙 Listen to your story",
                type="filepath",
                interactive=False,
                autoplay=True,
                elem_classes=["dv-player"],
            )
            gr.HTML(
                '<div class="dv-wave" aria-hidden="true">'
                + "".join("<span></span>" for _ in range(9))
                + "</div>"
                '<p class="dv-note">Told in your loved one\'s voice, just for you ✨</p>'
            )
            again_btn = gr.Button("Create Another Story 🔄", elem_classes=["dv-btn-accent"])

        tell_btn.click(
            fn=_run_story,
            inputs=[audio_input, genre_radio, mood_radio, hero_name_input,
                    language_radio, length_radio, energy_slider],
            outputs=[stage1, loading_panel, stage2, story_display, audio_player, error_box],
            api_name="tell_story",
        )

        again_btn.click(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
            ),
            inputs=[],
            outputs=[stage1, loading_panel, stage2, error_box],
        )

        gr.HTML(
            '<div class="dv-footer">'
            'DreamVoice &middot; English: VoxCPM2 &middot; ಕನ್ನಡ: IndicF5 &middot; Stories: MiniCPM'
            '</div>'
        )

    return demo


if __name__ == "__main__":
    demo = _build_app()
    demo.queue().launch(css=_stage_css(), server_port=7870)
