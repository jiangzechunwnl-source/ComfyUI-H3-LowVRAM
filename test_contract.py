"""Standard-library checks. Does not import torch, launch ComfyUI or use the network."""
import ast
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import types
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import run as launcher
import h3_low_vram as optimizer


class ContractTests(unittest.TestCase):
    def test_loopback_only(self):
        for value in ('127.0.0.1', '::1', 'localhost', b'localhost'):
            self.assertTrue(launcher.loopback(value))
        for value in ('0.0.0.0', '::', '192.168.1.2', 'example.com', '127.0.0.2', None):
            self.assertFalse(launcher.loopback(value))

    def test_network_audit_without_actual_network(self):
        sock = types.SimpleNamespace(family=socket.AF_INET)
        launcher.audit('socket.bind', (sock, ('127.0.0.1', 8188)))
        for event in ('socket.connect', 'socket.bind', 'socket.sendto', 'socket.sendmsg'):
            with self.assertRaises(PermissionError):
                launcher.audit(event, (sock, ('192.168.1.2', 8188)))
        with self.assertRaises(PermissionError):
            launcher.audit('socket.getaddrinfo', ('example.com', 443))
        with self.assertRaises(PermissionError):
            launcher.audit('subprocess.Popen', ('python',))

    def test_offline_environment(self):
        with patch.dict(launcher.os.environ, {}, clear=True):
            launcher.offline_environment(Path('local-cache'))
            for key in ('HF_HUB_OFFLINE', 'PIP_NO_INDEX', 'WANDB_DISABLED'):
                self.assertEqual(launcher.os.environ[key], '1')
            self.assertEqual(launcher.os.environ['HTTPS_PROXY'], 'http://127.0.0.1:9')
            self.assertIn('localhost', launcher.os.environ['NO_PROXY'])

    def test_compatibility_check_starts_no_server(self):
        fake = Path('test-local-comfy').resolve()
        output = io.StringIO()
        with patch.object(sys, 'argv', ['run.py', '--comfy-root', str(fake), '--check']), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(optimizer, '_verify') as verify, \
             patch.object(launcher.runpy, 'run_path') as execute, \
             contextlib.redirect_stdout(output):
            launcher.main()
        verify.assert_called_once()
        execute.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertFalse(result['torchImported'])
        self.assertFalse(result['serverStarted'])

    def test_mismatch_stops_before_launch(self):
        with patch.object(sys, 'argv', ['run.py', '--comfy-root', 'local-comfy']), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(optimizer, '_verify', side_effect=SystemExit('mismatch')), \
             patch.object(launcher.runpy, 'run_path') as execute, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                launcher.main()
            self.assertEqual(error.exception.code, 2)
            execute.assert_not_called()

    def test_import_order_guard(self):
        with patch.dict(sys.modules, {optimizer.TARGET: types.ModuleType(optimizer.TARGET)}):
            with self.assertRaises(SystemExit):
                optimizer.install()

    def test_package_is_english_and_syntax_valid(self):
        for path in PACKAGE.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            content = path.read_text(encoding='utf-8-sig')
            self.assertFalse(any('\u3400' <= c <= '\u9fff' for c in content), path.name)
            if path.suffix == '.py':
                ast.parse(content, filename=path.name)


if __name__ == '__main__':
    unittest.main(verbosity=2)

