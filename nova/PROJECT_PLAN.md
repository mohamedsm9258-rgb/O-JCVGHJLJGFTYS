# Project Plan: AI-Assisted 2D Anime Animation System

## Core Principle
> **Progressively construct animation frame by frame from Start State toward Target State, preserving character/scene consistency, with human creative control at every step.**

---

## Phase 0: Foundation (Week 1)

### Step 0.1: Project Scaffold
- **Goal**: Minimal Python package structure, config, logging, model cache
- **Files**: `pyproject.toml`, `src/nova/__init__.py`, `configs/base.yaml`, `scripts/download_models.py`
- **Test**: `python -m nova` prints version; models download to cache

### Step 0.2: Image I/O & Display
- **Goal**: Load Start/End images, display side-by-side in notebook
- **Files**: `src/nova/assets/images.py`, `notebooks/01_load_images.ipynb`
- **Test**: Load two images, show them, verify dimensions match

### Step 0.3: Configuration Schema
- **Goal**: Pydantic models for all user inputs
- **Files**: `src/nova/config/schemas.py`
- **Schema**: `AnimationConfig` (fps, frame_count, start_img, end_img, motion_prompt, audio_tracks[])

---

## Phase 1: Motion Planning MVP (Week 1-2)

### Step 1.1: Motion Prompt Parser (LLM-based)
- **Goal**: Convert natural language → structured keyframe plan
- **Files**: `src/nova/motion/planner.py`, `src/nova/motion/schemas.py`
- **Input**: "Raise right hand slowly over 2 seconds"
- **Output**: `MotionPlan` with keyframes: `[{frame: 0, pose: "hand_down"}, {frame: 24, pose: "hand_up"}, {frame: 48, pose: "hand_up"}]`
- **Test**: 5 diverse prompts → valid MotionPlan

### Step 1.2: Pose Representation
- **Goal**: Define pose format for 2D anime (skeleton + keypoints)
- **Files**: `src/nova/motion/pose.py`
- **Format**: 2D keypoints (head, shoulders, elbows, wrists, hips, knees, ankles) + optional IK targets
- **Test**: Serialize/deserialize pose, interpolate between two poses

### Step 1.3: Motion Interpolation
- **Goal**: Generate per-frame pose trajectory from keyframes
- **Files**: `src/nova/motion/interpolation.py`
- **Methods**: Linear, ease-in/out, custom curves per joint
- **Test**: Start pose → End pose over 48 frames → smooth trajectory

---

## Phase 2: Single Frame Generation (Week 2)

### Step 2.1: Frame Warping (CV-based)
- **Goal**: Warp Start image toward End pose using keypoint correspondences
- **Files**: `src/nova/generation/warp.py`
- **Method**: Sparse keypoints (Start→End) → Delaunay triangulation → affine warp per triangle
- **Output**: Warped frame + occlusion mask
- **Test**: Warp Start toward End pose → visual inspection

### Step 2.2: Generative Refinement (Inpainting)
- **Goal**: Fill occluded/new regions in warped frame
- **Files**: `src/nova/generation/refine.py`
- **Model**: SDXL Inpainting / Stable Diffusion Inpainting (configurable)
- **Input**: Warped frame + occlusion mask + prompt "anime style, consistent character"
- **Test**: Warped frame → refined frame → visual quality

### Step 2.3: Single Frame Pipeline
- **Goal**: End-to-end: Start frame + target pose → intermediate frame
- **Files**: `src/nova/generation/pipeline.py`
- **Flow**: `interpolate_pose → warp → refine → validate`
- **Test**: Frame 24 of "raise hand" sequence → acceptable quality

---

## Phase 3: Validation & Consistency (Week 2-3)

### Step 3.1: Consistency Metrics
- **Goal**: Quantitative checks per frame
- **Files**: `src/nova/validation/metrics.py`
- **Metrics**:
  - Identity: CLIP/DINOv2 similarity to Start frame
  - Structure: SSIM/LPIPS to previous frame
  - Pose accuracy: Keypoint distance to target pose
  - Artifact detection: Frequency analysis, anomaly score

### Step 3.2: Validation Gate
- **Goal**: Accept/Reject frame with specific failure reasons
- **Files**: `src/nova/validation/gate.py`
- **Thresholds**: Configurable per metric
- **Output**: `ValidationResult(passed, scores, failure_reasons, suggested_fixes)`
- **Test**: Known good/bad frames → correct classification

### Step 3.3: Auto-Retry Loop
- **Goal**: Regenerate with adjusted parameters on failure
- **Files**: `src/nova/generation/auto_retry.py`
- **Strategies**: Different seed, lower denoising strength, stronger IP-Adapter, pose correction
- **Max retries**: 3 per frame

---

## Phase 4: Sequence Generation (Week 3)

### Step 4.1: Frame Sequence Loop
- **Goal**: Generate full sequence Frame 1→N with validation
- **Files**: `src/nova/core/sequence.py`
- **State**: `AnimationState(start_frame, current_frame, target_frame, frame_history[])`
- **Loop**: For each frame: `plan_pose → generate → validate → accept/retry → store`
- **Test**: 12-frame sequence Start→End

### Step 4.2: Persistent Scene State
- **Goal**: Maintain character/environment identity across frames
- **Files**: `src/nova/core/scene_state.py`
- **Components**:
  - Identity embeddings (CLIP, DINOv2, ArcFace from Start frame)
  - Color palette histogram
  - Background reference (segmented from Start)
  - Style embedding (from Start frame)
- **Usage**: Injected into every generation + validation step

### Step 4.3: Start/Current/Target Awareness
- **Goal**: Generation conditioned on all three states
- **Files**: Modify `pipeline.py` to accept `scene_state`
- **Mechanism**: 
  - Start: Identity reference (IP-Adapter)
  - Current: Previous frame (warp source, latent init)
  - Target: End frame (pose target, IP-Adapter for final look)

---

## Phase 5: Human-in-the-Loop (Week 3-4)

### Step 5.1: Frame Inspection UI (Colab/Notebook)
- **Goal**: Display frames with onion-skinning, scrubber, acceptance buttons
- **Files**: `notebooks/02_inspect_frames.ipynb`, `src/nova/ui/colab_viewer.py`
- **Features**: 
  - Current frame + previous/next ghosted
  - Accept / Reject / Regenerate buttons
  - Pose overlay toggle
  - Metrics panel

### Step 5.2: User Correction Input
- **Goal**: Accept user guidance for rejected frame
- **Files**: `src/nova/interaction/corrections.py`
- **Input types**:
  - Text: "Hand higher, face unchanged"
  - Mask: User draws region to preserve/regenerate
  - Keypoint drag: Adjust target pose visually
- **Integration**: Correction → modified MotionPlan → regenerate from that frame

### Step 5.3: Localized Regeneration
- **Goal**: Regenerate from corrected frame forward (not full restart)
- **Files**: `src/nova/core/sequence.py` → `regenerate_from(frame_idx, correction)`
- **Test**: Reject frame 8 → correct → frames 8-12 regenerated, 1-7 unchanged

---

## Phase 6: Audio Integration (Week 4)

### Step 6.1: Audio Role Definition
- **Goal**: User defines role per track
- **Files**: `src/nova/audio/roles.py`
- **Roles**: `character_voice`, `background_music`, `sound_effects`, `reference_timing`, `custom`
- **Schema**: `AudioTrack(file, role, character_id?, weight?)`

### Step 6.2: Audio Feature Extraction (Role-Aware)
- **Goal**: Extract animation-relevant signals per role
- **Files**: `src/nova/audio/extract.py`
- **Per role**:
  - `character_voice`: WhisperX (phonemes, timestamps) + Crepe (pitch/prosody)
  - `background_music`: Librosa/Madmom (beats, onsets, tempo, chroma)
  - `sound_effects`: Onset detection + spectral classification
  - `reference_timing`: Beat/onset grid for motion timing

### Step 6.3: Audio → Motion Influence
- **Goal**: Convert audio features → motion plan adjustments
- **Files**: `src/nova/motion/audio_sync.py`
- **Mappings**:
  - Phonemes → mouth shapes (visemes)
  - Prosody (energy/pitch) → gesture intensity
  - Beats → keyframe timing snap
  - Onsets → sudden movements
- **Test**: Speech track → lip-sync keyframes; Music track → beat-synced gestures

---

## Phase 7: Dual Editing Modes (Week 4-5)

### Step 7.1: Timeline View (All Frames)
- **Goal**: See full sequence, add guidance anywhere
- **Files**: `notebooks/03_timeline_view.ipynb`
- **Features**: Frame strip, guidance layers, keyframe markers

### Step 7.2: Last N Frames View
- **Goal**: Focused editing on recent frames
- **Config**: `editing_window_size` (default 10)

### Step 7.3: Visual Annotation Tools
- **Goal**: Arrows, paths, masks, keypoint drags
- **Files**: `src/nova/interaction/annotations.py`
- **Storage**: Per-frame annotation layers → converted to motion corrections

---

## Phase 8: AI Conversation Interface (Week 5)

### Step 8.1: Chat → Motion Plan Translation
- **Goal**: Natural language → MotionPlan delta
- **Files**: `src/nova/interaction/chat.py`
- **Examples**:
  - "Raise hand more" → increase hand target Y in keyframes 10-20
  - "Don't change face" → add face preservation constraint
  - "Faster" → compress timing

### Step 8.2: Context-Aware Responses
- **Goal**: AI explains what it will change
- **Files**: `src/nova/interaction/chat.py`
- **Response**: "I'll adjust frames 10-18: hand target raised 20px, face locked to Start frame identity"

---

## Phase 9: Output & Polish (Week 5)  

### Step 9.1: Export Pipeline
- **Goal**: PNG sequence → MP4/WebM/GIF with audio sync
- **Files**: `src/nova/output/export.py`
- **FFmpeg**: Frame-accurate audio alignment

### Step 9.2: Metadata & Project Save
- **Goal**: Save/load full project state
- **Files**: `src/nova/core/project.py`
- **Format**: JSON + image assets + embeddings cache

---

## Dependency Graph

```
0.1 → 0.2 → 0.3
    ↓
1.1 → 1.2 → 1.3
    ↓
2.1 → 2.2 → 2.3
    ↓
3.1 → 3.2 → 3.3
    ↓
4.1 → 4.2 → 4.3
    ↓
5.1 → 5.2 → 5.3
    ↓
6.1 → 6.2 → 6.3
    ↓
7.1 → 7.2 → 7.3
    ↓
8.1 → 8.2
    ↓
9.1 → 9.2
```

---

## First Testable Prototype (End of Phase 2)

**Minimal MVP**: 
1. Load Start + End images
2. User types "raise right hand"
3. System generates Frame 24 (midpoint)
4. Display Start / Generated / End side-by-side
5. Manual visual verification

**Success Criteria**: Generated frame shows plausible intermediate hand position, character identity preserved, no major artifacts.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Warping breaks on large motion | Fallback: pure generative (img2img) for large deltas |
| Identity drift | Strong IP-Adapter from Start frame every frame; validation gate |
| Audio sync complexity | Phase 6 optional for MVP; ship visual-only first |
| Model selection paralysis | Start with SDXL Inpainting + OpenPose ControlNet; benchmark alternatives later |
| Colab → Cloud migration | Pure functions, no Colab-specific deps in core; inference behind interface |

---

## Next Immediate Action

**Step 0.1**: Create project scaffold. Confirm before proceeding.