#!/usr/bin/env python3
"""Unit tests for System-1 sidecar support in eval/harness/runner.py."""

import http.server
import json
import os
import threading
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import runner


class RunnerSystemOneTest(unittest.TestCase):
    def test_resolve_system1_config_defaults_and_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = runner.resolve_system1_config()
            self.assertFalse(cfg["enabled"])
            self.assertEqual(cfg["url"], "")
            self.assertEqual(cfg["model"], "bob-s1-0.8b")
            self.assertEqual(cfg["loop_guard_mode"], "hybrid")
            self.assertTrue(cfg["fallback_on_error"])

        with mock.patch.dict(
            os.environ,
            {
                "BOB_EVAL_SYSTEM1_URL": "http://127.0.0.1:8090/v1",
                "BOB_EVAL_SYSTEM1_MODEL": "bob-s1-custom",
                "BOB_EVAL_SYSTEM1_LOOP_GUARD_MODE": "shadow",
                "BOB_EVAL_SYSTEM1_FALLBACK_ON_ERROR": "false",
            },
            clear=True,
        ):
            cfg = runner.resolve_system1_config()
            self.assertTrue(cfg["enabled"])
            self.assertEqual(cfg["url"], "http://127.0.0.1:8090/v1")
            self.assertEqual(cfg["model"], "bob-s1-custom")
            self.assertEqual(cfg["loop_guard_mode"], "shadow")
            self.assertFalse(cfg["fallback_on_error"])

        # CLI args override environment.
        with mock.patch.dict(os.environ, {"BOB_EVAL_SYSTEM1_URL": "http://127.0.0.1:8090/v1"}, clear=True):
            cfg = runner.resolve_system1_config(
                system1_url="http://127.0.0.1:8099",
                system1_model="bob-s1-v2",
                system1_loop_guard_mode="enforce",
                system1_fallback_on_error="false",
            )
            self.assertEqual(cfg["url"], "http://127.0.0.1:8099")
            self.assertEqual(cfg["model"], "bob-s1-v2")
            self.assertEqual(cfg["loop_guard_mode"], "enforce")
            self.assertFalse(cfg["fallback_on_error"])

        with self.assertRaises(ValueError):
            runner.resolve_system1_config(system1_loop_guard_mode="invalid")

    def test_derive_system1_health_url_and_check_health(self):
        self.assertEqual(runner.derive_system1_health_url("http://127.0.0.1:8090"), "http://127.0.0.1:8090/healthz")
        self.assertEqual(runner.derive_system1_health_url("http://127.0.0.1:8090/v1"), "http://127.0.0.1:8090/healthz")
        self.assertEqual(runner.derive_system1_health_url("http://127.0.0.1:8090/v1/"), "http://127.0.0.1:8090/healthz")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/healthz":
                    body = json.dumps({"status": "ok", "ready": True, "model": "bob-s1-0.8b"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                return

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{srv.server_port}/v1"
            payload = runner.check_system1_health(url)
            self.assertEqual(payload["status"], "ok")
            self.assertTrue(payload["ready"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_build_config_yaml_and_cli_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            yaml_no_s1 = runner.build_config_yaml(
                "dummy", "http://127.0.0.1:15432/v1", "Qwen3.6", "", data_dir, 30,
                system1_cfg=runner.resolve_system1_config(),
            )
            self.assertNotIn("system1:", yaml_no_s1)

            s1_cfg = runner.resolve_system1_config(
                system1_url="http://127.0.0.1:8090/v1",
                system1_model="bob-s1-0.8b",
                system1_loop_guard_mode="hybrid",
                system1_fallback_on_error="false",
            )
            yaml_s1 = runner.build_config_yaml(
                "dummy", "http://127.0.0.1:15432/v1", "Qwen3.6", "", data_dir, 30,
                system1_cfg=s1_cfg,
            )
            self.assertIn("system1:", yaml_s1)
            self.assertIn('base_url: "http://127.0.0.1:8090/v1"', yaml_s1)
            self.assertIn('model: "bob-s1-0.8b"', yaml_s1)
            self.assertIn('loop_guard_mode: "hybrid"', yaml_s1)
            self.assertIn("fallback_on_error: false", yaml_s1)

        argv = [
            "runner.py", "manifest.json",
            "--system1-url=http://127.0.0.1:8090/v1",
            "--system1-fallback-on-error",
            "--limit", "4",
        ]
        self.assertEqual(runner._extract_cli_value(argv, "--system1-url"), "http://127.0.0.1:8090/v1")
        self.assertEqual(runner._extract_cli_bool(argv, "--system1-fallback-on-error"), "true")
        self.assertEqual(runner._extract_cli_value(argv, "--limit"), "4")

    def test_extract_and_summarize_telemetry(self):
        traj = {
            "events": [
                {
                    "event_type": "system1_decision",
                    "payload": {
                        "task": "compaction",
                        "calibration_applied": True,
                        "latency_ms": 14,
                        "applied_action": "skip",
                        "fallback_used": False,
                    },
                },
                {
                    "event_type": "compaction",
                    "payload": {
                        "should_compact": False,
                        "decision_source": "system1_skip",
                        "system1_latency_ms": 14,
                        "summarizer_latency_ms": 0,
                    },
                },
                {
                    "event_type": "system1_decision",
                    "payload": {
                        "task": "compaction",
                        "calibration_applied": True,
                        "latency_ms": 18,
                        "applied_action": "defer_to_system2",
                        "fallback_used": False,
                    },
                },
                {
                    "event_type": "compaction",
                    "payload": {
                        "should_compact": True,
                        "decision_source": "system2",
                        "system1_latency_ms": 18,
                        "summarizer_latency_ms": 620,
                    },
                },
                {
                    "event_type": "system1_decision",
                    "payload": {
                        "task": "loop_guard",
                        "calibration_applied": False,
                        "latency_ms": 9,
                        "applied_action": "warn",
                        "fallback_used": False,
                    },
                },
            ]
        }

        m = runner.extract_task_system1_metrics(traj)
        self.assertEqual(m["decision_events"], 3)
        self.assertEqual(m["compaction_gate_calls"], 2)
        self.assertEqual(m["compaction_calibration_applied"], 2)
        self.assertEqual(m["compaction_calibration_unapplied"], 0)
        self.assertEqual(m["compaction_system1_skipped_probes"], 1)
        self.assertEqual(m["compaction_deferred_to_system2"], 1)
        self.assertEqual(m["compaction_gate_latency_ms"], 32)
        self.assertEqual(m["compaction_summarizer_latency_ms"], 620)
        self.assertEqual(m["compaction_total_latency_ms"], 652)
        self.assertEqual(m["loop_guard_calls"], 1)
        self.assertEqual(m["loop_guard_interventions"], 1)
        self.assertEqual(m["fallback_count"], 0)
        self.assertEqual(m["error_count"], 0)

        batch = runner.summarize_batch_metrics([
            {
                "task_id": "t1",
                "benchmark": "frames",
                "grading_status": "graded",
                "benchmark_reward": 1.0,
                "system1_metrics": m,
            },
            {
                "task_id": "t2",
                "benchmark": "swe-bench-lite",
                "grading_status": "graded",
                "benchmark_reward": 0.0,
                "patch_generated": True,
                "resolved": False,
                "trajectory": {"events": []},
            },
        ])
        self.assertEqual(batch["per_benchmark"]["frames"]["passed"], 1)
        self.assertEqual(batch["per_benchmark"]["swe-bench-lite"]["swe_patched"], 1)
        self.assertEqual(batch["system1_metrics"]["compaction_total_latency_ms"], 652)


if __name__ == "__main__":
    unittest.main()
