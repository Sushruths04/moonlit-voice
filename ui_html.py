"""Gradio HTML blocks for the DreamVoice UI — starfield, moon, and storybook."""

from __future__ import annotations

import gradio as gr

_STARFIELD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=Pacifico&display=swap');

/* ── reset & canvas ──────────────────────────────────────────── */
#dv-stars-root, #dv-stars-root * { box-sizing: border-box; margin: 0; padding: 0; }
#dv-stars-root {
  position: fixed; inset: 0; z-index: -1; overflow: hidden;
  background: linear-gradient(170deg, #0a0a1a 0%, #1a1a3e 100%);
  pointer-events: none;
}

/* ── stars ───────────────────────────────────────────────────── */
#dv-stars-root .star {
  position: absolute; border-radius: 50%; background: #fff;
  animation: dv-twinkle var(--dur) ease-in-out infinite alternate;
  opacity: 0;
}
@keyframes dv-twinkle { 0% { opacity: .15; } 100% { opacity: 1; } }

/* ── moon ────────────────────────────────────────────────────── */
#dv-moon {
  position: absolute; top: 60px; right: 10%; width: 90px; height: 90px;
  border-radius: 50%; background: radial-gradient(circle at 35% 35%, #fffbe6 0%, #f5c842 60%, #c9a020 100%);
  box-shadow: 0 0 40px 12px rgba(245,200,66,.35), 0 0 80px 24px rgba(245,200,66,.15);
  animation: dv-moon-glow 4s ease-in-out infinite alternate;
}
@keyframes dv-moon-glow {
  0% { box-shadow: 0 0 40px 12px rgba(245,200,66,.35), 0 0 80px 24px rgba(245,200,66,.15); }
  100% { box-shadow: 0 0 55px 18px rgba(245,200,66,.45), 0 0 100px 32px rgba(245,200,66,.2); }
}

/* ── storybook flip container ────────────────────────────────── */
#dv-storybook-wrap {
  perspective: 1200px; display: flex; justify-content: center; align-items: center;
  min-height: 340px; margin: 0 auto 1.5rem; max-width: 680px;
}
#dv-storybook {
  width: 100%; min-height: 340px; position: relative;
  transform-style: preserve-3d;
  transition: transform 0.8s cubic-bezier(.4,0,.2,1);
  transform-origin: center left;
  /* Plays each time Stage 2 becomes visible (display none→block restarts it),
     so the book "opens" on every reveal. */
  animation: dv-book-open .85s cubic-bezier(.2,.7,.2,1) both;
}
@keyframes dv-book-open {
  0%   { transform: rotateY(-72deg); opacity: 0; }
  100% { transform: rotateY(0deg);   opacity: 1; }
}
#dv-storybook.dv-flipped { transform: rotateY(180deg); }

#dv-book-front, #dv-book-back {
  position: absolute; inset: 0; backface-visibility: hidden;
  border-radius: 24px; padding: 2rem 2.2rem;
  font-family: 'Lora', Georgia, serif; font-size: 1.05rem; line-height: 1.7;
  color: #eee; overflow-y: auto;
  background: linear-gradient(135deg, rgba(30,30,70,.85) 0%, rgba(15,15,45,.92) 100%);
  border: 1.5px solid rgba(245,200,66,.25);
  box-shadow: 0 8px 32px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
}
#dv-book-back { transform: rotateY(180deg); }

#dv-storybook h2.dv-story-title {
  font-family: 'Pacifico', cursive; font-size: 1.65rem; color: #f5c842;
  margin-bottom: 1rem; text-shadow: 0 2px 8px rgba(245,200,66,.25);
}
#dv-storybook .dv-story-text { white-space: pre-wrap; }

/* ── accent buttons (shared) ─────────────────────────────────── */
.dv-btn-accent {
  background: #f5c842; color: #1a1a3e; border: none; border-radius: 24px;
  padding: .65rem 1.8rem; font-weight: 700; font-size: 1rem; cursor: pointer;
  font-family: 'Lora', Georgia, serif; transition: transform .15s, box-shadow .15s;
}
.dv-btn-accent:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(245,200,66,.4); }
.dv-btn-accent:active { transform: translateY(0); }

/* ── stage headings & helpers ────────────────────────────────── */
.dv-section-title {
  font-family: 'Pacifico', cursive; font-size: 1.5rem; color: #f5c842;
  text-align: center; margin: .5rem 0 1rem;
}
.dv-note {
  font-family: 'Lora', Georgia, serif; font-size: .88rem;
  color: rgba(255,255,255,.55); text-align: center; margin-top: .3rem;
}
"""

_STARFIELD_JS = """
(function(){
  var root = document.getElementById('dv-stars-root');
  if (!root) return;
  for (var i = 0; i < 120; i++){
    var s = document.createElement('div');
    s.className = 'star';
    var size = 1.2 + Math.random() * 2.5;
    s.style.width  = size + 'px';
    s.style.height = size + 'px';
    s.style.left   = Math.random() * 100 + '%';
    s.style.top    = Math.random() * 100 + '%';
    s.style.setProperty('--dur', (1.5 + Math.random() * 3) + 's');
    s.style.animationDelay = (Math.random() * 4) + 's';
    root.appendChild(s);
  }
})();
"""


def starfield_background() -> gr.HTML:
    """Full-viewport animated starfield with a glowing moon."""
    return gr.HTML(
        value='<div id="dv-stars-root"></div><div id="dv-moon"></div>',
        head=f"<style>{_STARFIELD_CSS}</style>",
        js_on_load=_STARFIELD_JS,
        elem_id="dv-stars-root",
    )


def storybook_markup(story: str = "", title: str = "Tonight's Story") -> str:
    """Return the storybook HTML string with *story* and *title* HTML-escaped.

    Use this both to build the initial component and to update it after a story
    is generated, so the storybook frame/title/escaping are preserved.
    """
    import html as _html

    safe_title = _html.escape(title)
    safe_story = _html.escape(story) if story else "<em>The storybook is waiting…</em>"

    return f"""
<div id="dv-storybook-wrap">
  <div id="dv-storybook">
    <div id="dv-book-front">
      <h2 class="dv-story-title">{safe_title}</h2>
      <div class="dv-story-text">{safe_story}</div>
    </div>
    <div id="dv-book-back">
      <h2 class="dv-story-title">The End</h2>
      <p style="text-align:center;margin-top:2rem;color:#f5c842;font-size:1.2rem;">
        Sweet dreams ✨
      </p>
    </div>
  </div>
</div>
"""


def storybook_html(story: str = "", title: str = "Tonight's Story") -> gr.HTML:
    """Render the story inside the 3D storybook flip container."""
    return gr.HTML(value=storybook_markup(story, title), elem_id="dv-storybook")
