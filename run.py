"""Experimental, offline launcher for a fingerprint-checked native ComfyUI H3.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

VERSION = "0.1.0"


def loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host) in (ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1"))
    except (ValueError, TypeError):
        return False


def audit(event, args):
    # Install before importing ComfyUI/torch. The launcher never starts another process.
    if event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"):
        import socket
        sock, address = args[0], args[-1]
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address is not None and not loopback(address[0]):
            raise PermissionError("H3 LowVRAM: only local loopback network traffic is allowed")
    elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
        if not loopback(args[0]):
            raise PermissionError("H3 LowVRAM: external DNS is disabled")
    elif event == "socket.getnameinfo" and not loopback(args[0][0]):
        raise PermissionError("H3 LowVRAM: external DNS is disabled")
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        raise PermissionError("H3 LowVRAM: this isolated launcher does not permit child processes")


def offline_environment(cache):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "DIFFUSERS_OFFLINE", "MODELSCOPE_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "WANDB_DISABLED", "PIP_NO_INDEX", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"):
        os.environ[name] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["WANDB_MODE"] = "offline"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[name] = "http://127.0.0.1:9"
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = "127.0.0.1,localhost,::1"
    for name, subdir in (("HF_HOME", "huggingface"), ("TORCHINDUCTOR_CACHE_DIR", "torch"),
                         ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
        os.environ[name] = str(cache / subdir)


def main():
    parser = argparse.ArgumentParser(description="Offline native H3 chunked inference; no model or dependency downloads.")
    parser.add_argument("--comfy-root", required=True, type=Path, help="Existing local ComfyUI folder")
    parser.add_argument("--extra-packages", type=Path, help="Optional existing local H3 Python package overlay")
    parser.add_argument("--models-root", type=Path, help="Optional existing local models directory")
    parser.add_argument("--workspace", type=Path, help="Optional separate folder for input, output, temp and user data")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--check", action="store_true", help="Check source compatibility without importing torch or starting a server")
    options = parser.parse_args()
    if not 1 <= options.port <= 65535:
        parser.error("port must be between 1 and 65535")
    for path in (options.comfy_root, options.extra_packages, options.models_root, options.workspace):
        if path is not None and (str(path).startswith(("\\\\", "//")) or "://" in str(path)):
            parser.error("only local filesystem paths are accepted")
    comfy = options.comfy_root.resolve()
    model_source = comfy / "comfy/ldm/minimax/model.py"
    if not (comfy / "main.py").is_file() or not model_source.is_file():
        parser.error("local native H3 ComfyUI source is missing; nothing will be downloaded")
    if options.extra_packages is not None and not options.extra_packages.is_dir():
        parser.error("local extra-packages folder does not exist")
    if options.models_root is not None and not options.models_root.is_dir():
        parser.error("local models-root folder does not exist")
    if options.models_root is not None:
        options.models_root = options.models_root.resolve()
    if options.workspace is not None:
        options.workspace = options.workspace.resolve()
    # Windows embedded Python may omit the script directory from sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import h3_low_vram
    try:
        h3_low_vram._verify(SimpleNamespace(__file__=str(model_source)))
    except SystemExit:
        parser.error("unsupported H3 source fingerprint; use a verified compatible checkout, or wait for an updated release")
    if options.check:
        print(json.dumps({"version": VERSION, "compatibleSource": True, "chunkTokens": h3_low_vram.ROWS,
                          "torchImported": "torch" in sys.modules, "serverStarted": False}))
        return
    # Discard another ComfyUI checkout baked into a portable interpreter's search path.
    sys.path[:] = [entry for entry in sys.path if not (
        Path(entry).name.casefold() == "comfyui" and Path(entry).resolve() != comfy)]
    sys.path.insert(0, str(comfy))
    if options.extra_packages:
        sys.path.insert(0, str(options.extra_packages.resolve()))
    sys.dont_write_bytecode = True
    offline_environment(comfy / ".h3-low-vram-cache")
    sys.addaudithook(audit)
    if importlib.util.find_spec("torch") is None:
        parser.error("local torch is missing; use your existing ComfyUI Python environment")
    os.chdir(comfy)
    sys.argv = [str(comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(options.port),
                "--disable-api-nodes", "--disable-auto-launch", "--disable-all-custom-nodes",
                "--database-url", "sqlite:///:memory:", "--lowvram", "--cache-none", "--reserve-vram", "1"]
    if options.models_root:
        sys.argv.extend(["--models-directory", str(options.models_root.resolve())])
    if options.workspace:
        workspace = options.workspace.resolve()
        for option, name in (("--input-directory", "input"), ("--output-directory", "output"),
                             ("--temp-directory", "temp"), ("--user-directory", "user")):
            directory = workspace / name
            directory.mkdir(parents=True, exist_ok=True)
            sys.argv.extend([option, str(directory)])
    h3_low_vram.install()
    print("H3 LowVRAM 0.1.0: source checked; 1024-token chunks; exact full-K/V attention; experimental.", flush=True)
    runpy.run_path(str(comfy / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
