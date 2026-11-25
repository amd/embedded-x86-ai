# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The Gemma intent node - language as a ROS 2 citizen.

This node closes the loop between component 1 and component 2: transcripts
arrive as ``std_msgs/String`` on ``/vla/transcript`` (from the speech stack,
``ros2 topic pub``, or the notebook), and parsed intents leave as JSON on
``/vla/intent`` - so ANY node can react to language without linking against
the LLM.

Two design points carry over from the real pipeline, and both are yours to
implement:

- **The stop fast path.** "stop"/"halt"/"freeze" is keyword-matched on the
  raw transcript and published IMMEDIATELY - the safety stop must never wait
  in line behind a language model.
- **QoS asymmetry.** Camera frames were best-effort (a lost frame is free);
  intents are commands, so this node uses default RELIABLE QoS - a lost
  "stop" is not free.

Notebook: ``notebooks/02_ros2/03_intent_node.ipynb``
Run it for real (llama-server running, or ``--mock`` for the keyword brain):
    python -m ros.intent_node [--mock]
Then, in another terminal:
    ros2 topic pub --once /vla/transcript std_msgs/String "{data: pick up the ball}"
    ros2 topic echo /vla/intent
"""

from __future__ import annotations

import argparse
import json
import logging

from common.config import load_config
from common.intents import Intent, ParsedCommand, is_noise, is_stop

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS2_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]


class IntentNode(Node):
    """Subscribes to transcripts, publishes grammar-shaped intent JSON.

    ``parser`` is injected: any callable ``str -> ParsedCommand``. On the rig
    that's :meth:`models.llm.LlamaIntentParser.parse` (Gemma 4 on the iGPU);
    in tests it's a keyword stand-in. The node doesn't know or care - which
    is exactly why it's a clean seam.
    """

    def __init__(self, cfg: dict, parser):
        super().__init__("llama_intent_node")
        n = cfg.get("intent_node", {}) or {}
        self.transcript_topic = n.get("transcript_topic", "/vla/transcript")
        self.intent_topic = n.get("intent_topic", "/vla/intent")
        self._parse = parser
        self._pub = None
        self._setup_ros()
        self.get_logger().info(
            f"intent node: {self.transcript_topic} → {self.intent_topic}"
        )

    def _setup_ros(self) -> None:
        """Wire the transcript subscription and the intent publisher.

        Both are ``std_msgs/String``. Use RELIABLE delivery (the rclpy
        default - just pass a queue depth like ``10``): intents are
        commands, not a sensor stream, and must not be dropped.
        """
        # >>> TODO 2.10: IntentNode._setup_ros - notebooks/02_ros2/03_intent_node.ipynb
        self.create_subscription(String, self.transcript_topic, self._on_transcript, 10)
        self._pub = self.create_publisher(String, self.intent_topic, 10)
        # <<< TODO 2.10

    def _on_transcript(self, msg) -> None:
        """Route one transcript: stop fast path → noise gate → LLM → publish.

        1. :func:`common.intents.is_stop` on the RAW text → publish
           ``{"command": "stop", "object": ""}`` immediately and return -
           no LLM round trip on the safety path.
        2. :func:`common.intents.is_noise` → drop silently (Whisper's
           silence hallucinations must not become robot commands).
        3. Otherwise ``self._parse(text)`` and publish the result via
           :meth:`publish_intent`.
        """
        text = msg.data or ""
        # >>> TODO 2.11: IntentNode._on_transcript - notebooks/02_ros2/03_intent_node.ipynb
        if is_stop(text):
            self.publish_intent(ParsedCommand(Intent.STOP, transcript=text))
            return
        if is_noise(text):
            self.get_logger().info(f"ignoring noise transcript: {text!r}")
            return
        cmd = self._parse(text)
        self.publish_intent(cmd)
        # <<< TODO 2.11

    def publish_intent(self, cmd: ParsedCommand) -> None:
        """PROVIDED: serialize a ParsedCommand as the grammar-shaped JSON."""
        payload = json.dumps(
            {"command": cmd.intent.value, "object": cmd.object_name or ""}
        )
        self._pub.publish(String(data=payload))
        self.get_logger().info(f"intent: {payload}")


def main() -> None:
    """CLI: bring up the intent node (mock or Gemma-backed) and spin until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if not ROS2_AVAILABLE:
        raise SystemExit(
            "rclpy not importable - source ROS 2 before running "
            "(offline checks: python -m ros.selftest)"
        )
    parser = argparse.ArgumentParser(description="Gemma intent node")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="keyword stand-in brain (no llama-server needed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        from common.fixtures import offline_parse_text

        def parse(text: str) -> ParsedCommand:
            """Mock intent brain: keyword-route ``text`` into a ParsedCommand (no LLM)."""
            d = offline_parse_text(text)
            params = {"object": d["object"]} if d.get("object") else {}
            return ParsedCommand(
                Intent.from_string(d["command"]), params=params, transcript=text
            )

        print("(mock brain - keyword routing, no LLM)")
    else:
        from models.llm import LlamaIntentParser

        parse = LlamaIntentParser(cfg).parse

    rclpy.init()
    node = IntentNode(cfg, parse)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
