"""
app.py — Real-Time Video-to-String using Qwen2.5-VL-3B-Instruct
Hardware target : Acer Nitro V15 · RTX GPU · 6 GB VRAM · CUDA 12.1
Memory strategy : 4-bit NF4 quantization via bitsandbytes + device_map="auto"
"""

import sys
import time
import threading
import textwrap

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GPU VALIDATION
#     Abort early if no CUDA device is found so the user gets a clear message
#     instead of a confusing CUDA / CPU mismatch later.
# ─────────────────────────────────────────────────────────────────────────────

def validate_gpu() -> None:
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA-capable GPU detected. "
              "Make sure the NVIDIA driver and CUDA 12.1 toolkit are installed.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free_vram   = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / (1024 ** 3)

    print("=" * 60)
    print(f"  GPU  : {device_name}")
    print(f"  VRAM : {total_vram:.1f} GB total  |  {free_vram:.1f} GB available")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL + PROCESSOR LOADING
#     BitsAndBytesConfig enables NF4 4-bit quantization:
#       • load_in_4bit          – cuts VRAM usage by ~4× vs FP32
#       • bnb_4bit_quant_type   – NF4 is the best quality 4-bit format
#       • bnb_4bit_compute_dtype– internal compute stays in BF16 for speed
#       • bnb_4bit_use_double_quant – second quantization of scale factors
#                                    (saves ~0.4 bits/param extra)
#     device_map="auto" lets Accelerate decide which layers go to GPU/CPU
#     so we never exceed 6 GB VRAM.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Vision token budget — non-negotiable for 6 GB VRAM:
#   min_pixels = 256 × 28 × 28  ≈ 200 K pixels  (keeps small frames cheap)
#   max_pixels = 448 × 28 × 28  ≈ 351 K pixels  (hard ceiling to avoid OOM)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 448 * 28 * 28


def load_model_and_processor():
    print("[INFO] Loading model …  (first run downloads ~2 GB — be patient)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # best quality 4-bit format
        bnb_4bit_compute_dtype=torch.bfloat16,  # fast BF16 matmuls on Ampere+
        bnb_4bit_use_double_quant=True,      # nested quant for extra VRAM savings
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",          # Accelerate auto-distributes layers
        torch_dtype=torch.bfloat16, # base dtype before quantization
    )
    model.eval()

    # min/max_pixels are passed to the processor; it will resize every image
    # so it never exceeds MAX_PIXELS vision tokens.
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("[INFO] Model ready.\n")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INFERENCE  (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_PROMPT = (
    "Describe what you see in this image in one concise sentence. "
    "Focus on the main subject, their action, and the environment."
)
MAX_NEW_TOKENS = 96   # keep short → faster; adjust if you want more detail


def run_inference(frame_bgr: np.ndarray, model, processor) -> str:
    """Convert one BGR OpenCV frame to a description string."""
    # OpenCV → PIL (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": INFERENCE_PROMPT},
            ],
        }
    ]

    # Build the chat prompt text
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # process_vision_info handles image resizing within [MIN_PIXELS, MAX_PIXELS]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move input tensors to the same device as the first model parameter
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy — deterministic & faster
        )

    # Strip the prompt tokens; decode only the newly generated part
    prompt_len = inputs["input_ids"].shape[1]
    new_ids    = generated_ids[:, prompt_len:]
    description = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    return description


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OVERLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_text_overlay(frame: np.ndarray, text: str, fps: float) -> np.ndarray:
    """Draw a semi-transparent bottom banner with the AI description."""
    h, w = frame.shape[:2]

    # Wrap long text to ~60 chars per line
    wrapped = textwrap.wrap(text, width=60) if text else ["Waiting for first inference…"]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    line_h     = 22
    padding    = 8

    banner_h = padding * 2 + line_h * len(wrapped)
    overlay  = frame.copy()

    # Dark semi-transparent rectangle
    cv2.rectangle(overlay, (0, h - banner_h - 4), (w, h), (0, 0, 0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Text lines
    for i, line in enumerate(wrapped):
        y = h - banner_h - 4 + padding + (i + 1) * line_h
        cv2.putText(frame, line, (padding, y),
                    font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)

    # FPS counter top-right
    fps_str = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_str, font, 0.45, 1)
    cv2.putText(frame, fps_str, (w - tw - 8, 18),
                font, 0.45, (80, 255, 80), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_INTERVAL = 1.75   # seconds between AI calls (tune: 1.5 – 2.0)


def main() -> None:
    validate_gpu()
    model, processor = load_model_and_processor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Could not open webcam (index 0).")
        sys.exit(1)

    # State shared between main thread and inference thread
    latest_description: list[str] = ["Initializing…"]
    inference_lock   = threading.Lock()
    inference_running = threading.Event()   # set → thread is busy

    def inference_worker(frame_snapshot: np.ndarray) -> None:
        try:
            result = run_inference(frame_snapshot, model, processor)
            with inference_lock:
                latest_description[0] = result
            print(f"[AI] {result}")
        except Exception as exc:
            print(f"[ERROR] Inference failed: {exc}")
        finally:
            inference_running.clear()   # signal: thread finished

    last_inference_time = 0.0
    fps_timer = time.perf_counter()
    frame_count = 0
    display_fps = 0.0

    print("[INFO] Webcam open. Press  q  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame — retrying …")
            continue

        now = time.perf_counter()

        # ── Non-blocking inference trigger ──────────────────────────────────
        # Fire a background thread if:
        #   • no thread is currently running (inference_running not set)
        #   • enough time has elapsed since the last call
        if not inference_running.is_set() and (now - last_inference_time) >= INFERENCE_INTERVAL:
            last_inference_time = now
            inference_running.set()
            # Pass a *copy* of the frame so the main loop can keep reading
            t = threading.Thread(
                target=inference_worker,
                args=(frame.copy(),),
                daemon=True,
            )
            t.start()

        # ── FPS calculation ─────────────────────────────────────────────────
        frame_count += 1
        elapsed = now - fps_timer
        if elapsed >= 0.5:
            display_fps = frame_count / elapsed
            frame_count = 0
            fps_timer   = now

        # ── Overlay ─────────────────────────────────────────────────────────
        with inference_lock:
            current_text = latest_description[0]

        frame = draw_text_overlay(frame, current_text, display_fps)

        cv2.imshow("Video-to-String  |  Qwen2.5-VL-3B  (press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ─────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()