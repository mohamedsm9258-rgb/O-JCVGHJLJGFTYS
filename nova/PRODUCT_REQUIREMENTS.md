# Product Requirements: AI-Assisted 2D Anime Animation System

## 1. Product Vision

An interactive AI system that constructs 2D anime animations frame-by-frame from a Start Image to an End Image, with human creative control at every step. The system behaves like an AI animation assistant that understands motion, preserves consistency, and responds to natural language and visual guidance.

---

## 2. Core User Flow

```
Start Image + End Image + Motion Description + Audio Tracks
         ↓
    Motion Planning (LLM → Structured Keyframes)
         ↓
    Frame-by-Frame Generation Loop
         ├─→ Warp previous frame toward target pose
         ├─→ Generative refinement (inpaint)
         ├─→ Validate (identity, structure, pose, artifacts)
         ├─→ Accept / Reject / Regenerate
         └─→ Human correction (text, mask, keypoint drag)
         ↓
    Continue from accepted frame
         ↓
    Export: Frame sequence + synced audio
```

---

## 3. Functional Requirements

### 3.1 Input Specification
| Input | Format | Required |
|-------|--------|----------|
| Start Image | PNG/JPG/WEBP, any resolution | Yes |
| End Image | PNG/JPG/WEBP, same resolution as Start | Yes |
| Motion Description | Natural language text | Yes |
| Frame Count / FPS / Duration | Integer frames + FPS (12/24/30) | Yes |
| Audio Tracks | 0-N files (WAV/MP3) + role per track | Optional |
| Audio Roles | Enum: `character_voice`, `background_music`, `sound_effects`, `reference_timing`, `custom` | Per track |

### 3.2 Motion Planning
- **FR-MP-01**: Convert natural language → structured keyframe plan with poses and timing
- **FR-MP-02**: Support easing curves (linear, ease-in, ease-out, custom) per joint
- **FR-MP-03**: Allow manual keyframe editing (add/remove/modify poses)
- **FR-MP-04**: Audio-driven adjustments (phonemes→visemes, beats→keyframe snap, prosody→intensity)

### 3.3 Frame Generation
- **FR-FG-01**: Hybrid warp + generative refinement pipeline
- **FR-FG-02**: Previous accepted frame as primary warp source
- **FR-FG-03**: Start frame as identity reference (appearance, style)
- **FR-FG-04**: End frame as target reference (final pose, look)
- **FR-FG-05**: Persistent scene state (identity embeddings, palette, background) injected every frame
- **FR-FG-06**: Configurable generation model (SDXL Inpainting default, swappable)

### 3.4 Validation & Quality Control
- **FR-VQ-01**: Per-frame metrics: identity (CLIP/DINO/ArcFace), structure (SSIM/LPIPS), pose accuracy, artifacts
- **FR-VQ-02**: Configurable thresholds per metric
- **FR-VQ-03**: Auto-retry with parameter adjustment (max 3 attempts)
- **FR-VQ-04**: Explicit failure reasons for human review

### 3.5 Human-in-the-Loop
- **FR-HL-01**: Frame-by-frame inspection with onion-skinning (prev/next ghosted)
- **FR-HL-02**: Accept / Reject / Regenerate per frame
- **FR-HL-03**: Correction inputs: text instruction, preservation mask, keypoint drag
- **FR-HL-04**: Localized regeneration from corrected frame forward (no full restart)
- **FR-HL-05**: Two editing modes: Full Timeline + Last N Frames (configurable N)

### 3.6 Audio Integration
- **FR-AU-01**: User-defined role per audio track (no assumptions)
- **FR-AU-02**: Role-appropriate feature extraction
- **FR-AU-03**: Audio features → motion plan influence (configurable mappings)
- **FR-AU-04**: Frame-accurate audio-visual sync in export

### 3.7 Conversation Interface
- **FR-CI-01**: Natural language → MotionPlan delta
- **FR-CI-02**: AI explains planned changes before applying
- **FR-CI-03**: Constraints understood ("don't change face", "faster", "loop last 10 frames")

### 3.8 Output
- **FR-OUT-01**: PNG frame sequence
- **FR-OUT-02**: MP4/WebM/GIF with audio
- **FR-OUT-03**: Project save/load (full state + assets)
- **FR-OUT-04**: Metadata JSON per frame (pose, metrics, generation params)

---

## 4. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| **Visual Domain** | 2D Anime (primary), extensible to other 2D styles |
| **Resolution** | 512×512 to 2048×2048 (configurable) |
| **Frame Rate** | 12, 24, 30 FPS (configurable) |
| **Sequence Length** | 12–300+ frames |
| **Identity Consistency** | CLIP similarity >0.85 to Start frame across sequence |
| **Temporal Coherence** | SSIM >0.90 between adjacent frames |
| **Generation Time** | Draft: <2s/frame (512px), Final: <10s/frame (1024px) on A100 |
| **VRAM** | ≤24GB for 1024px draft mode |
| **Deployment** | Colab (dev) → Google Cloud Vertex AI (prod) |
| **Interface** | Notebook/Colab (MVP) → Web UI (React/Three.js) later |
| **Dependencies** | Open-source models only; no proprietary APIs in core |

---

## 5. User Roles

| Role | Capabilities |
|------|--------------|
| **Animator/Director** | Full control: motion planning, frame review, corrections, conversation |
| **Technical Artist** | Model config, pipeline tuning, custom ControlNet/IP-Adapter |
| **Pipeline TD** | Batch processing, API integration, cloud deployment |

---

## 6. Out of Scope (MVP)

- 3D animation / rigging
- Automatic in-betweening without Start/End images
- Text-to-video generation
- Real-time playback during generation
- Multi-user collaboration
- Custom model training (LoRA per project is Phase 2)
- Dedicated dubbing system (architecture-ready only)

---

## 7. Acceptance Criteria for MVP

1. **Start→End interpolation**: 12-frame sequence from Start to End with natural motion
2. **Identity preservation**: Character recognizable and consistent across all frames
3. **Human correction**: User can reject frame 6, draw "hand higher", system regenerates 6–12
4. **Audio sync**: Character voice track → lip-sync visemes on correct frames
5. **Conversation**: "Make it slower" → timing stretched, frames regenerated
6. **Export**: MP4 with synced audio plays correctly

---

## 8. Technical Constraints

- **Models**: SDXL base, SDXL Inpainting, ControlNet (OpenPose/Depth/Canny), IP-Adapter Plus
- **CV**: MediaPipe/RTMPose (keypoints), RAFT (flow), DepthAnything v2 (depth), SAM (segmentation)
- **Audio**: WhisperX (ASR+alignment), Crepe (pitch), Librosa/Madmom (music)
- **LLM**: Local (Llama-3/Ollama) or API for motion planning + chat
- **Framework**: PyTorch 2.3+, Python 3.10+, FastAPI for service layer
- **Storage**: Local filesystem (Colab) → GCS (Cloud)

---

## 9. Success Metrics

| Metric | Target |
|--------|--------|
| Frame acceptance rate (auto) | >80% |
| Frames requiring human correction | <20% |
| Identity drift (CLIP @ frame 48) | <0.05 drop from frame 1 |
| User task completion (5-sec animation) | <30 min |
| Export quality (subjective) | "Production usable" by animator |