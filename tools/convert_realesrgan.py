#!/usr/bin/env python3
"""Real-ESRGAN の公開重みを Core ML (.mlpackage) に変換する。

Core ML の image 入出力を使うので、Swift 側は CVPixelBuffer をそのまま渡せる。
入力は固定解像度（タイルサイズ）で、フレーム全体は Swift 側でタイル分割して流す。

例:
    python3 convert_realesrgan.py                      # general-x4v3 / タイル 256
    python3 convert_realesrgan.py --model x4plus --tile 192
    python3 convert_realesrgan.py --model x2plus --tile 384
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

from realesrgan_arch import RRDBNet, SRVGGNetCompact

BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"

# 名前 -> (重みURL, ネットワーク生成関数, 拡大倍率, 説明)
MODELS = {
    "general-x4v3": (
        f"{BASE}/v0.2.5.0/realesr-general-x4v3.pth",
        lambda: SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4),
        4,
        "軽量・汎用。動画向けの既定",
    ),
    "animevideo-x4v3": (
        f"{BASE}/v0.2.5.0/realesr-animevideov3.pth",
        lambda: SRVGGNetCompact(num_feat=64, num_conv=16, upscale=4),
        4,
        "アニメ動画向け。最速",
    ),
    "x4plus": (
        f"{BASE}/v0.1.0/RealESRGAN_x4plus.pth",
        lambda: RRDBNet(num_block=23, scale=4),
        4,
        "実写向け高画質。重い",
    ),
    "x4plus-anime": (
        f"{BASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        lambda: RRDBNet(num_block=6, scale=4),
        4,
        "イラスト・アニメ向け高画質",
    ),
    "x2plus": (
        f"{BASE}/v0.2.1/RealESRGAN_x2plus.pth",
        lambda: RRDBNet(num_block=23, scale=2),
        2,
        "実写向け高画質の 2 倍版。720p 素材に",
    ),
}


class ImageIOWrapper(nn.Module):
    """Core ML の image 入出力に合わせるための薄いラッパー。

    入力は ImageType の scale=1/255 で 0-1 に正規化済み。出力は 0-255 に戻して
    そのまま画像として取り出せるようにする。
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0) * 255.0


def download(url: str, destination: Path) -> Path:
    if destination.exists():
        print(f"重みは取得済み: {destination.name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"重みをダウンロード中: {url}")
    with urllib.request.urlopen(url) as response, open(destination, "wb") as file:
        file.write(response.read())
    print(f"  -> {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def load_compact_weights(path: Path):
    """ローカルの SRVGG Compact 重みを、形状から構成を推定して読み込む。

    コミュニティ配布のアニメ向けモデル (AnimeJaNai / AniSD など) は
    Real-ESRGAN Compact と同じ構造・同じキー名 (body.N.*) で配布されている。
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if key in state:
            state = state[key]
            break
    if "body.0.weight" not in state:
        raise SystemExit(f"SRVGG Compact 構造ではありません (body.0.weight がない): {path}")
    convs = [k for k in state if k.endswith(".weight") and state[k].dim() == 4]
    num_feat = state[convs[0]].shape[0]
    out_channels = state[convs[-1]].shape[0]
    scale = int(round((out_channels / 3) ** 0.5))
    num_conv = len(convs) - 2
    print(f"構成を推定: num_feat={num_feat}, num_conv={num_conv}, x{scale}")
    network = SRVGGNetCompact(num_feat=num_feat, num_conv=num_conv, upscale=scale)
    network.load_state_dict(state, strict=True)
    return network.eval(), scale


def load_network(name: str, weights: Path) -> nn.Module:
    _, factory, _, _ = MODELS[name]
    network = factory()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    # 公開重みは params_ema / params のどちらかに包まれている。
    for key in ("params_ema", "params"):
        if key in state:
            state = state[key]
            break
    network.load_state_dict(state, strict=True)
    return network.eval()


def verify(mlmodel, torch_model: nn.Module, tile: int) -> None:
    """同じ入力を PyTorch と Core ML に通し、ずれが実用範囲か確認する。"""
    from PIL import Image

    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(tile, tile, 3), dtype=np.uint8)

    # torch_model は ImageIOWrapper なので、戻り値はすでに 0-255 スケール。
    with torch.no_grad():
        reference = torch_model(torch.from_numpy(pixels).permute(2, 0, 1)[None].float() / 255.0)
    reference = reference.squeeze(0).permute(1, 2, 0).numpy().round()

    predicted = np.array(mlmodel.predict({"input": Image.fromarray(pixels)})["output"].convert("RGB"), dtype=np.float32)

    mae = float(np.abs(reference - predicted).mean())
    mse = float(((reference - predicted) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(f"検証: 平均絶対誤差 {mae:.3f} / 255, PSNR {psnr:.1f} dB")
    if mae > 2.0:
        print("  警告: 誤差が大きめです。FP16 の丸めにしては外れているかもしれません。", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="general-x4v3",
                        help="変換するモデル (既定: general-x4v3)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="ローカルの .pth を変換する (SRVGG Compact 構造を自動判別。--model は無視)")
    parser.add_argument("--tile", type=int, default=256,
                        help="モデルが受け取る正方タイルの一辺。8 の倍数 (既定: 256)")
    parser.add_argument("--output", type=Path, default=None, help="出力先 .mlpackage")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).parent / "weights")
    parser.add_argument("--skip-verify", action="store_true", help="PyTorch との出力比較を省略する")
    parser.add_argument("--list", action="store_true", help="使えるモデルを一覧表示して終了")
    arguments = parser.parse_args()

    if arguments.list:
        print("使えるモデル:")
        for name, (_, _, scale, note) in sorted(MODELS.items()):
            print(f"  {name:<18} x{scale}  {note}")
        return 0

    if arguments.tile % 8 != 0:
        parser.error("--tile は 8 の倍数で指定してください")

    import coremltools as ct

    if arguments.weights is not None:
        network, scale = load_compact_weights(arguments.weights)
        model_name = arguments.weights.stem
    else:
        url, _, scale, _ = MODELS[arguments.model]
        weights = download(url, arguments.weights_dir / Path(url).name)
        network = load_network(arguments.model, weights)
        model_name = arguments.model

    parameters = sum(p.numel() for p in network.parameters())
    print(f"モデル: {model_name} (x{scale}, {parameters / 1e6:.2f}M パラメータ)")

    wrapper = ImageIOWrapper(network).eval()
    example = torch.rand(1, 3, arguments.tile, arguments.tile)
    print(f"トレース中: 入力 {arguments.tile}x{arguments.tile} -> 出力 {arguments.tile * scale}x{arguments.tile * scale}")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    output = arguments.output or (
        Path(__file__).resolve().parent.parent / "Models" / f"{model_name}-t{arguments.tile}.mlpackage"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Core ML へ変換中…")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="input", shape=example.shape, scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        compute_precision=ct.precision.FLOAT16 if arguments.precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = f"{model_name} x{scale} ({arguments.tile}px タイル)"
    mlmodel.author = "Real-ESRGAN (xinntao) / Core ML 変換"
    mlmodel.license = "BSD-3-Clause"

    if output.exists():
        import shutil
        shutil.rmtree(output)
    mlmodel.save(str(output))
    print(f"保存しました: {output}")

    if not arguments.skip_verify:
        verify(mlmodel, wrapper, arguments.tile)

    print(f"\n使い方:\n  swift run upscale <入力動画> --model {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
