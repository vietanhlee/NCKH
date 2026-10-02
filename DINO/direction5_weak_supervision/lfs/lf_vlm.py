"""
=============================================================================
 Hướng B: Weak Supervision — LF3: VLM Zero-Shot Labeling Function
 Phân loại mức độ ùn tắc qua Vision-Language Models (Qwen2-VL / InternVL)
 Kiểm tra đồng thuận giữa 2 prompts đảo thứ tự để tự động Abstain khi không chắc
=============================================================================
"""

import json
from typing import Dict, Optional, Tuple, Any


class VLMLabelingFunction:
    """
    LF3: Sử dụng VLM zero-shot với giao thức kiểm định đồng thuận kép (Dual-Prompt Consensus).
    """

    PROMPT_A = """You are a traffic analyst. Look at this CCTV image of an urban road in Vietnam,
where most vehicles are motorbikes. Classify the congestion on the ROAD SURFACE:
0 = free flow: large gaps between vehicles
1 = moderate: dense but clear gaps, vehicles moving
2 = slow: vehicles close together, small gaps
3 = jammed: road almost fully covered, vehicles not moving
If the image is too dark, blurry or blocked to judge, answer "unknown".
Reply ONLY with JSON: {"level": 0|1|2|3|"unknown"}"""

    PROMPT_B = """You are a traffic analyst inspecting traffic congestion on Vietnam urban streets.
Evaluate the vehicles on the asphalt road:
3 = jammed: road completely congested, motionless vehicles
2 = slow: tightly packed vehicles moving at low speed
1 = moderate: continuous flow with safe spacing
0 = free flow: completely open road
If unclear, answer "unknown".
Reply ONLY with JSON: {"level": 0|1|2|3|"unknown"}"""

    def parse_response(self, text_resp: str) -> Optional[int]:
        try:
            # Tìm kiếm JSON
            text_resp = text_resp.strip()
            start = text_resp.find("{")
            end = text_resp.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text_resp[start:end])
                lvl = data.get("level")
                if lvl in [0, 1, 2, 3, "0", "1", "2", "3"]:
                    return int(lvl)
        except Exception:
            pass
        return None

    def __call__(
        self,
        response_a: str,
        response_b: Optional[str] = None
    ) -> Tuple[int, float]:
        """
        Nếu chỉ có 1 prompt: parse trực tiếp.
        Nếu có cả 2 prompt: kiểm tra đồng thuận. Nếu lệch nhau -> Abstain (-1).
        """
        ans_a = self.parse_response(response_a)
        if ans_a is None:
            return -1, 0.0

        if response_b is not None:
            ans_b = self.parse_response(response_b)
            if ans_b is None or ans_b != ans_a:
                # 2 prompt không đồng thuận -> Abstain
                return -1, 0.0

        return ans_a, 1.0
