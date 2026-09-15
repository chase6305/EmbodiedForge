# EmbodiedForge README workflow diagram

- Asset: `embodiedforge-overview.png`
- Generated: 2026-09-15
- Tool: built-in `image_gen` (imagegen skill); no CLI fallback.
- Purpose: shared educational workflow diagram for the Chinese and English README. It describes native Go1/H1 training, evaluation and recorded-motion replay, with separate notes for external SDK workflows and live Go1 control.
- Style reference: the local `chase6305.github.io/content/posts/ai/transformer-attention/assets` collection, especially `training-lifecycle.webp` and `transformer-block-overview.webp`: white background, pastel modules, line icons and directional arrows. Reference files were inspected; their artwork is not bundled here.
- Scope: the browser panel is schematic. Native H1 is an initial training implementation; the diagram does not claim a qualified walking policy. Robot motion replay supports MuJoCo, OpenGL and OVRTX. Raster is available for core point tasks, not robot mesh replay.

## Initial generation prompt

```text
Use case: infographic-diagram.
Asset type: explanatory workflow diagram for the English and Chinese GitHub README of EmbodiedForge.
Create a precise educational engineering diagram, landscape 1792x1024 or similar. White background, delicate blue/purple/orange/green outlines, very pale pastel panel fills, flat line-art icons, crisp dark sans-serif labels, clean dark directional arrows, ample whitespace. Match the style of textbook Transformer explanation diagrams: logical modules, small meaningful icons, clear visual reading order. No 3D concept art, photoreal robots, neon, decorative hero scene or fake screenshot.

Title at top left: "EmbodiedForge"
Subtitle: "Train, evaluate, and replay robot policies"

MAIN ROW: four evenly spaced tall panels connected left to right with three arrows. Each panel has a colored numbered circle and a clear heading. Use exactly these labels:
Blue panel 1 heading "Robot + Task", with a simple outlined quadruped and humanoid icon, and three short labels below:
"MJCF assets"
"Go1 / H1"
"Commands + rewards"

Purple panel 2 heading "Native Training", with a 2x3 grid of small simulation tiles feeding a tiny neural-network icon. Labels:
"MuJoCo + mjbatch"
"PyTorch PPO"
"CPU environments"

Orange panel 3 heading "Evaluate + Record", with an outlined checkpoint document, small ruler/checklist icon (not a success checkmark), and motion frames. Labels:
"Checkpoint"
"Tracking + survival"
"Motion NPZ"

Green panel 4 heading "Web Replay", with a modest flat browser icon showing a robot stick figure on a grid and play/pause symbols, then labels:
"Go1 / H1 motion"
"Raster / MuJoCo"
"OpenGL / OVRTX"

BELOW main row: a thin purple return arrow from bottom of the Checkpoint area back to Native Training, labeled "Resume".
BOTTOM: two compact side-by-side neutral outlined notes, visually secondary. Left note heading "Independent SDK workflows", body "Microduck / IsaacLab H1 / Wuji / MPC". Right note heading "Live Web control", body "Go1 policy + velocity commands".
These bottom notes are standalone scope notes; NO arrows from them to main row, no implication all workflows share the same runtime.
Final tiny but readable footer: "Native Go1 / H1 training does not require IsaacLab."

Accuracy constraints: this depicts supported workflows, not an actual user interface, not a performance claim. H1 training is an initial implementation; never imply a qualified walking policy. No GPU training label, no real-hardware deployment, no H1 live control. Main arrows indicate workflow progression, not data training on rendered images. Keep every quoted English label exact, no extra text. Use large readable typography and restrained flat icons; avoid clutter.
```

## Accuracy correction prompt

Applied to the generated diagram with the built-in editing tool:

```text
Edit this EmbodiedForge workflow diagram with only two accuracy corrections. Preserve all layout, dimensions, colors, typography, arrows, headings, other labels and icons.
1. In purple Native Training panel, all six small simulation tiles must show a quadruped on a FLAT level grid floor. Remove all steps, blocks, cylinders, ramps and obstacles. Each tile is a simple flat-floor environment with one quadruped.
2. In green Web Replay panel replace the exact text "Raster / MuJoCo" with "MuJoCo". Keep the next line "OpenGL / OVRTX" unchanged. Robot motion replay does not support Raster.
Everything else must remain unchanged and exact.
```
