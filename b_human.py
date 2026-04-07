"""
unified_ai_pipeline.py — Real-Time Video Perception-to-Reasoning with Human-in-the-Loop
Hardware target : Acer Nitro V15 · RTX GPU · 6 GB VRAM · CUDA 12.1
Memory strategy : 4-bit NF4 quantization via bitsandbytes + device_map="auto"
LLM inference : Ollama llama3.2:3b (CPU)
Hardware : LinkBrain ESP32 with Light on PIN 4

Pipeline: Video Stream → Vision Model → Generated Text → LLM Workflow → Threat Decision → Human Approval → PIN 4 Light Control
"""

import sys
import time
import threading
import textwrap
import uuid
import asyncio
from typing import Annotated, Literal

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

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict

# LinkBrain imports for hardware control
import nest_asyncio
nest_asyncio.apply()

from linkbrain import ESP32Controller, Light
from linkbrain_core.llm.gemini import GeminiProvider
from linkbrain_core.prompts.template import PromptBuilder, DeviceContext
from linkbrain_core.parsers.action_parser import ActionParser
from linkbrain_core.tools import ToolRegistry
from linkbrain_core.tools.light import LightTool


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GPU VALIDATION
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
# 2.  VISION MODEL (Qwen2.5-VL) LOADING
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 448 * 28 * 28


def load_vision_model():
    print("[INFO] Loading Qwen2.5-VL model … (first run downloads ~2 GB)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("[INFO] Vision model ready.\n")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VISION INFERENCE (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_PROMPT = (
    "Describe what you see in this image in one concise sentence. "
    "Focus on the main subject, their action, and the environment."
)
MAX_NEW_TOKENS = 96


def run_vision_inference(frame_bgr: np.ndarray, model, processor) -> str:
    """Convert one BGR OpenCV frame to a description string."""
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

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_ids    = generated_ids[:, prompt_len:]
    description = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    return description


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LINKBRAIN HARDWARE INTEGRATION (THREAT → PIN 4 LIGHT CONTROL)
# ─────────────────────────────────────────────────────────────────────────────

class LinkBrainHardware:
    """Singleton manager for ESP32 hardware control"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not LinkBrainHardware._initialized:
            self.controller = None
            self.threat_light = None
            self.is_connected = False
            LinkBrainHardware._initialized = True
    
    def initialize(self):
        """Initialize ESP32 connection and light on PIN 4"""
        print("\n[LINKBRAIN] Initializing hardware connection...")
        
        try:
            # Connect to ESP32 via Bluetooth
            self.controller = ESP32Controller(
                mode="bluetooth",
                device_address="3A51DF0E-1520-9120-DA2D-C48E2F714E30"
            )
            self.controller.connect()
            self.is_connected = True
            
            # Initialize light on PIN 4
            self.threat_light = Light("threat_indicator", self.controller, pin=4)
            
            # Ensure light starts OFF
            self.threat_light.off()
            
            print(f"[LINKBRAIN] ✓ Connected to ESP32")
            print(f"[LINKBRAIN] ✓ Threat indicator light initialized on PIN 4")
            print(f"[LINKBRAIN] ✓ Initial state: OFF\n")
            
            return True
            
        except Exception as e:
            print(f"[LINKBRAIN] ✗ Connection failed: {e}")
            print(f"[LINKBRAIN] Continuing without hardware control...\n")
            self.is_connected = False
            return False
    
    def light_on(self):
        """Turn ON PIN 4 light"""
        if not self.is_connected:
            return
        try:
            self.threat_light.on()
            print(f"[LINKBRAIN] 🔴 PIN 4 Light: ON")
        except Exception as e:
            print(f"[LINKBRAIN] ✗ Failed to turn ON light: {e}")
    
    def light_off(self):
        """Turn OFF PIN 4 light"""
        if not self.is_connected:
            return
        try:
            self.threat_light.off()
            print(f"[LINKBRAIN] 🟢 PIN 4 Light: OFF")
        except Exception as e:
            print(f"[LINKBRAIN] ✗ Failed to turn OFF light: {e}")
    
    def cleanup(self):
        """Clean shutdown: turn off light and disconnect"""
        if self.is_connected and self.threat_light:
            try:
                self.threat_light.off()
                self.controller.disconnect()
                print(f"[LINKBRAIN] ✓ Cleanup complete - Light OFF, disconnected")
            except:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LLM WORKFLOW (LangGraph with Ollama)
# ─────────────────────────────────────────────────────────────────────────────

class TacticalState(TypedDict):
    messages:      Annotated[list, add_messages]
    vision_string: str
    threat_found:  bool
    analysis:      str
    final_report:  str
    action_taken:  bool
    human_decision: str  # 'approve' or 'deny'


ANALYZE_PROMPT = """You are a tactical AI threat-assessment system.

Analyze the following visual scene description from drone footage.
Determine if it contains HIGH-RISK elements such as:
- Weapons, firearms, explosives
- Active conflict or violence
- Humans in danger or under threat
- Critical infrastructure under attack
- Hazardous materials or events

Vision string:
\"\"\"{vision}\"\"\"

Respond in EXACTLY this format (nothing else):
THREAT: YES or NO
REASON: <one concise sentence explaining your decision>"""


class LLMWorkflow:
    """Wrapper for LangGraph workflow that processes vision text automatically"""
    
    def __init__(self, hardware_handler=None):
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0.2,
            num_predict=1024,
        )
        self.memory = MemorySaver()
        self.hardware_handler = hardware_handler
        self.graph = self._build_graph()
        self.current_thread_id = None
        self.last_result = None
        self.pending_threat = False
        self.human_decision_made = threading.Event()
        self.human_decision = None
    
    def _build_graph(self):
        builder = StateGraph(TacticalState)
        
        builder.add_node("analyzer", self._node_analyzer)
        builder.add_node("human_check", self._node_human_check)
        builder.add_node("action", self._node_action)
        builder.add_node("deny_report", self._node_deny_report)
        
        builder.set_entry_point("analyzer")
        builder.add_conditional_edges("analyzer", self._route_after_analysis)
        builder.add_conditional_edges("human_check", self._route_after_human)
        builder.add_edge("action", END)
        builder.add_edge("deny_report", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def _node_analyzer(self, state: TacticalState) -> dict:
        prompt = ANALYZE_PROMPT.format(vision=state["vision_string"])
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        raw = response.content.strip()
        threat_found = "THREAT: YES" in raw.upper()
        reason_line = next(
            (ln for ln in raw.splitlines() if ln.upper().startswith("REASON:")),
            "REASON: (see raw output)"
        )
        reason = reason_line.split(":", 1)[-1].strip()
        
        return {
            "messages": [response],
            "threat_found": threat_found,
            "analysis": reason,
        }
    
    def _node_human_check(self, state: TacticalState) -> dict:
        """Wait for human decision (Y/N)"""
        print(f"\n{'='*50}")
        print(f"⚠ HUMAN DECISION REQUIRED ⚠")
        print(f"{'='*50}")
        print(f"Threat Analysis: {state['analysis']}")
        print(f"\nPress 'Y' to APPROVE (Turn OFF light)")
        print(f"Press 'N' to DENY (Turn ON light)")
        print(f"{'='*50}\n")
        
        # Reset decision event
        self.human_decision_made.clear()
        self.pending_threat = True
        
        # Wait for human decision (with timeout to prevent infinite wait)
        timeout = 30  # 30 seconds timeout
        if not self.human_decision_made.wait(timeout=timeout):
            print(f"[TIMEOUT] No decision made within {timeout}s. Defaulting to DENY (light ON)")
            self.human_decision = "DENY"
        
        decision_text = "APPROVE" if self.human_decision == "APPROVE" else "DENY"
        print(f"\n[HUMAN] Decision: {decision_text}")
        
        # Control light based on human decision
        if self.hardware_handler:
            if self.human_decision == "APPROVE":
                # Approve action - Turn OFF light
                self.hardware_handler.light_off()
                print(f"[LINKBRAIN] Human APPROVED - Light OFF")
            else:
                # Deny action - Turn ON light
                self.hardware_handler.light_on()
                print(f"[LINKBRAIN] Human DENIED - Light ON")
        
        self.pending_threat = False
        return {"human_decision": self.human_decision}
    
    def _node_action(self, state: TacticalState) -> dict:
        print("\n[ACTION] ✓ Action APPROVED by human")
        return {
            "action_taken": True,
            "messages": [AIMessage(content="ACTION APPROVED")],
        }
    
    def _node_deny_report(self, state: TacticalState) -> dict:
        print("\n[ACTION] ✗ Action DENIED by human - Light remains ON")
        return {
            "final_report": "Human denied the action. Light remains ON as warning.",
            "messages": [AIMessage(content="ACTION DENIED")],
        }
    
    def _route_after_analysis(self, state: TacticalState) -> Literal["human_check", "__end__"]:
        """Route to human checkpoint only when a threat is found"""
        if state["threat_found"]:
            print("[ROUTER] Threat detected! Waiting for human decision...")
            return "human_check"
        else:
            print("[ROUTER] No threat detected.")
            if self.hardware_handler:
                self.hardware_handler.light_off()
            return END
    
    def _route_after_human(self, state: TacticalState) -> Literal["action", "deny_report"]:
        """Route based on human decision"""
        if state.get("human_decision") == "APPROVE":
            return "action"
        return "deny_report"
    
    def set_human_decision(self, decision: str):
        """Called from main thread to set human decision"""
        if self.pending_threat:
            self.human_decision = decision
            self.human_decision_made.set()
    
    def process_vision_text(self, vision_text: str) -> dict:
        """Main entry point: process vision text through the workflow"""
        self.current_thread_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": self.current_thread_id}}
        
        initial_state: TacticalState = {
            "messages": [],
            "vision_string": vision_text,
            "threat_found": False,
            "analysis": "",
            "final_report": "",
            "action_taken": False,
            "human_decision": "",
        }
        
        try:
            result = self.graph.invoke(initial_state, config=thread_config)
            self.last_result = result
            return result
        except Exception as exc:
            print(f"[ERROR] LLM Workflow failed: {exc}")
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  OVERLAY HELPERS WITH HUMAN-IN-THE-LOOP UI
# ─────────────────────────────────────────────────────────────────────────────

def draw_text_overlay(frame: np.ndarray, vision_text: str, llm_result: dict, 
                      threat_status: bool, pending_threat: bool, fps: float) -> np.ndarray:
    """Draw overlay with vision output and human-in-the-loop UI"""
    h, w = frame.shape[:2]
    
    # Vision text (wrapped)
    vision_wrapped = textwrap.wrap(f"Vision: {vision_text}", width=55) if vision_text else ["Waiting for vision..."]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_h = 22
    padding = 8
    
    # Calculate banner height
    base_lines = len(vision_wrapped) + 2  # status + separator
    extra_lines = 0
    
    # Add human-in-the-loop UI if threat is pending
    if pending_threat:
        extra_lines = 4  # Y/N prompt lines
        status = "⚠ THREAT DETECTED - HUMAN DECISION REQUIRED"
        status_color = (80, 80, 255)
    elif llm_result and "threat_found" in llm_result:
        if llm_result["threat_found"]:
            status = "⚠ THREAT - Waiting for human input"
            status_color = (80, 80, 255)
        else:
            status = "✓ NO THREAT - Safe"
            status_color = (80, 255, 80)
    else:
        status = "Analyzing..."
        status_color = (220, 220, 220)
    
    total_lines = base_lines + extra_lines
    banner_h = padding * 2 + line_h * total_lines
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h - 4), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    
    y_offset = h - banner_h - 4 + padding
    
    # Status line
    cv2.putText(frame, status, (padding, y_offset),
                font, font_scale, status_color, thickness, cv2.LINE_AA)
    y_offset += line_h
    
    # Separator
    cv2.putText(frame, "-" * 50, (padding, y_offset - 5),
                font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    
    # Vision text lines
    for line in vision_wrapped:
        cv2.putText(frame, line, (padding, y_offset),
                    font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        y_offset += line_h
    
    # Human-in-the-loop UI (only when threat detected)
    if pending_threat:
        y_offset += 5
        cv2.putText(frame, "═" * 55, (padding, y_offset - 5),
                    font, 0.4, (80, 80, 255), 1, cv2.LINE_AA)
        
        # Y/N prompt with visual styling
        cv2.putText(frame, "⚠ HUMAN DECISION REQUIRED ⚠", (padding + 50, y_offset),
                    font, 0.65, (80, 80, 255), 2, cv2.LINE_AA)
        y_offset += line_h + 5
        
        cv2.putText(frame, "Press 'Y' → APPROVE ACTION (Turn OFF Light)", (padding + 30, y_offset),
                    font, 0.55, (80, 255, 80), 1, cv2.LINE_AA)
        y_offset += line_h
        
        cv2.putText(frame, "Press 'N' → DENY ACTION (Keep Light ON)", (padding + 30, y_offset),
                    font, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
        y_offset += line_h
        
        cv2.putText(frame, "═" * 55, (padding, y_offset),
                    font, 0.4, (80, 80, 255), 1, cv2.LINE_AA)
    
    # FPS counter
    fps_str = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_str, font, 0.45, 1)
    cv2.putText(frame, fps_str, (w - tw - 8, 18),
                font, 0.45, (80, 255, 80), 1, cv2.LINE_AA)
    
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE (Video → Vision → LLM → Human → Hardware)
# ─────────────────────────────────────────────────────────────────────────────

class VideoPerceptionPipeline:
    """Main pipeline orchestrating video capture, vision inference, LLM, human, and hardware"""
    
    def __init__(self, inference_interval: float = 1.75):
        self.inference_interval = inference_interval
        self.vision_model = None
        self.vision_processor = None
        self.llm_workflow = None
        self.hardware = None
        
        # Threading state
        self.current_vision_text = "Initializing system..."
        self.current_llm_result = {}
        self.current_threat_status = False
        self.pending_threat = False
        self.vision_lock = threading.Lock()
        self.vision_running = threading.Event()
        
        # FPS tracking
        self.fps_timer = time.perf_counter()
        self.frame_count = 0
        self.display_fps = 0.0
        
        # Vision inference queue
        self.last_inference_time = 0.0
    
    def initialize(self):
        """Load vision model, LLM workflow, and hardware"""
        print("\n[INIT] Loading AI pipeline with Human-in-the-Loop...\n")
        
        # Initialize LinkBrain hardware first
        self.hardware = LinkBrainHardware()
        self.hardware.initialize()
        
        # Load vision model (GPU)
        self.vision_model, self.vision_processor = load_vision_model()
        
        # Load LLM workflow with hardware handler reference
        print("[INFO] Initializing LLM workflow...")
        self.llm_workflow = LLMWorkflow(hardware_handler=self.hardware)
        print("[INFO] LLM workflow ready.\n")
        
        print("[INIT] Pipeline ready. Starting video capture...\n")
    
    def _vision_inference_worker(self, frame_snapshot: np.ndarray):
        """Background thread for vision model inference"""
        try:
            result = run_vision_inference(frame_snapshot, self.vision_model, self.vision_processor)
            print(f"\n[VISION] {result}")
            
            # Update current vision text
            with self.vision_lock:
                self.current_vision_text = result
            
            # Automatically send to LLM workflow
            print("[LLM] Processing vision output...")
            llm_result = self.llm_workflow.process_vision_text(result)
            
            # Store LLM result and threat status
            with self.vision_lock:
                self.current_llm_result = llm_result
                self.current_threat_status = llm_result.get("threat_found", False)
                # Check if we're waiting for human decision
                self.pending_threat = self.llm_workflow.pending_threat
            
            # Print LLM decision
            if "threat_found" in llm_result:
                if llm_result["threat_found"]:
                    print(f"[LLM] ⚠ THREAT DETECTED - Awaiting human decision (Y/N)")
                else:
                    print(f"[LLM] ✓ No threat detected - Safe")
                
        except Exception as exc:
            print(f"[ERROR] Vision inference failed: {exc}")
            with self.vision_lock:
                self.current_vision_text = f"Inference error: {exc}"
        finally:
            self.vision_running.clear()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: trigger inference and handle human input"""
        now = time.perf_counter()
        
        # Check for human input (Y/N) - this runs every frame
        # Use a small wait time to not block the loop
        key = cv2.waitKey(1) & 0xFF
        
        # Handle human decision when threat is pending
        if self.pending_threat:
            if key == ord('y') or key == ord('Y'):
                print("\n[HUMAN] User pressed 'Y' - APPROVING action (Turning OFF light)")
                self.llm_workflow.set_human_decision("APPROVE")
                with self.vision_lock:
                    self.pending_threat = False
            elif key == ord('n') or key == ord('N'):
                print("\n[HUMAN] User pressed 'N' - DENYING action (Keeping light ON)")
                self.llm_workflow.set_human_decision("DENY")
                with self.vision_lock:
                    self.pending_threat = False
        
        # Trigger vision inference (non-blocking)
        if not self.vision_running.is_set() and (now - self.last_inference_time) >= self.inference_interval:
            self.last_inference_time = now
            self.vision_running.set()
            
            t = threading.Thread(
                target=self._vision_inference_worker,
                args=(frame.copy(),),
                daemon=True,
            )
            t.start()
        
        # Update FPS
        self.frame_count += 1
        elapsed = now - self.fps_timer
        if elapsed >= 0.5:
            self.display_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = now
        
        # Draw overlay with current state
        with self.vision_lock:
            vision_text = self.current_vision_text
            llm_result = self.current_llm_result.copy()
            threat_status = self.current_threat_status
            pending = self.pending_threat
        
        output_frame = draw_text_overlay(frame, vision_text, llm_result, threat_status, pending, self.display_fps)
        return output_frame
    
    def run(self):
        """Main loop: capture video, process frames, display output"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[FATAL] Could not open webcam (index 0).")
            sys.exit(1)
        
        print("[INFO] Webcam open. Press 'q' to quit.\n")
        print("=" * 60)
        print("PIPELINE ACTIVE: Video → Vision → LLM → Human Decision → PIN 4 Light")
        print("=" * 60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Dropped frame — retrying...")
                    continue
                
                processed_frame = self.process_frame(frame)
                cv2.imshow("AI Pipeline | Human-in-the-Loop | Y=Approve N=Deny Q=Quit", processed_frame)
                
                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Clean shutdown - ensure light is OFF
            if self.hardware:
                self.hardware.cleanup()
            cap.release()
            cv2.destroyAllWindows()
            print("\n[INFO] Pipeline stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    validate_gpu()
    
    pipeline = VideoPerceptionPipeline(inference_interval=1.75)
    pipeline.initialize()
    pipeline.run()


if __name__ == "__main__":
    main()