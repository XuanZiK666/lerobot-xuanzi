#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

# 关键：确保优先导入当前仓库源码（/home/woan/lerobot/src）
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action


def _resolve_model_dir(model_path: str) -> Path:
    """支持两种输入：
    1) .../checkpoints/030000
    2) .../checkpoints/030000/pretrained_model
    """
    p = Path(model_path).expanduser().resolve()
    if (p / "pretrained_model").is_dir():
        return p / "pretrained_model"
    return p


def _load_train_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "train_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tensor_or_array_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _to_hwc_uint8(image: Any) -> np.ndarray:
    """把图片转成 HWC + uint8，匹配 prepare_observation_for_inference 的输入约定。"""
    arr = _tensor_or_array_to_numpy(image)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            maxv = float(np.max(arr)) if arr.size > 0 else 0.0
            if maxv <= 1.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _pick_device(device_arg: str | None, fallback_from_cfg: str | None) -> torch.device:
    def _safe_torch_device(try_device: str) -> torch.device:
        try_device = str(try_device)
        if try_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用")
            return torch.device(try_device)
        if try_device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS 不可用")
            return torch.device("mps")
        if try_device == "xpu":
            if not torch.xpu.is_available():
                raise RuntimeError("XPU 不可用")
            return torch.device("xpu")
        if try_device == "cpu":
            return torch.device("cpu")
        return torch.device(try_device)

    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    if device_arg is not None:
        return _safe_torch_device(device_arg)

    if fallback_from_cfg is not None:
        try:
            return _safe_torch_device(fallback_from_cfg)
        except Exception:
            pass

    return _auto_device()


def _render_action_panel(
    actions: np.ndarray,
    height: int,
    width: int,
    current_step: int,
    action_labels: list[str],
) -> np.ndarray:
    """渲染动作曲线面板（RGB uint8）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_w = max(320, int(width))
    safe_h = max(240, int(height))

    # 以 100dpi 近似匹配输出像素尺寸
    fig, ax = plt.subplots(figsize=(safe_w / 100.0, safe_h / 100.0), dpi=100)
    x = np.arange(current_step + 1, dtype=np.int32)
    hist = actions[: current_step + 1]

    for d in range(hist.shape[1]):
        label = action_labels[d] if d < len(action_labels) else f"a{d}"
        ax.plot(x, hist[:, d], linewidth=1.8, label=label)

    # 全局 x 范围固定到整个推理长度，便于观察进度
    ax.set_xlim(0, max(actions.shape[0] - 1, 1))

    # y 轴用当前历史值自动缩放并加边距
    y_min = float(hist.min())
    y_max = float(hist.max())
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(f"Predicted action curves (step {current_step + 1}/{actions.shape[0]})")
    ax.set_xlabel("Frame step")
    ax.set_ylabel("Action value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()

    fig.canvas.draw()
    panel_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    panel_rgb = panel_rgba[:, :, :3].copy()
    plt.close(fig)
    return panel_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 LeRobot 训练好的策略，对一个 episode 的视频逐帧模拟推理。")
    parser.add_argument(
        "--model-path",
        type=str,
        default="output_lerobot_train/act/checkpoints/030000",
        help="checkpoint 路径或 pretrained_model 路径",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="数据集 repo_id，例如 TommyZihao/lerobot_zihao_dataset_a。默认从 train_config.json 自动读取",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="数据集本地根目录。默认从 train_config.json 自动读取",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="要模拟的 episode 编号")
    parser.add_argument("--max-frames", type=int, default=150, help="最多推理多少帧")
    parser.add_argument("--device", type=str, default=None, help="推理设备：cuda/cpu/mps/xpu")
    parser.add_argument(
        "--save-actions",
        type=str,
        default="",
        help="可选：保存动作到 .npy 文件（例如 outputs/episode0_actions.npy）",
    )
    parser.add_argument(
        "--export-compare-video",
        type=str,
        default="",
        help="可选：导出对比视频（左：原视频帧，右：预测动作曲线），例如 outputs/ep0_compare.mp4",
    )
    parser.add_argument(
        "--compare-panel-width",
        type=int,
        default=960,
        help="对比视频中动作曲线面板宽度（像素）",
    )
    parser.add_argument(
        "--compare-fps",
        type=float,
        default=0.0,
        help="导出视频帧率；<=0 时自动使用数据集 fps",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_dir = _resolve_model_dir(args.model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_dir}")

    train_cfg = _load_train_config(model_dir)
    repo_id = args.repo_id or train_cfg.get("dataset", {}).get("repo_id")
    dataset_root = args.dataset_root or train_cfg.get("dataset", {}).get("root")

    if repo_id is None:
        raise ValueError("缺少 --repo-id，且 train_config.json 中也没有 dataset.repo_id")

    # 1) 加载 policy
    policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(model_dir)

    # 2) 设备选择
    device = _pick_device(args.device, policy_cfg.device)
    policy.to(device)
    policy.eval()

    # 3) 加载 pre/post processor（使用训练时保存的配置）
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=str(model_dir))

    # 4) 加载目标 episode（视频帧会按时间戳从 mp4 解码）
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[args.episode_index],
        download_videos=False,
    )

    image_keys = list(policy_cfg.image_features.keys())
    if len(image_keys) == 0:
        raise ValueError("当前策略没有视觉输入，不能做视频推理模拟")

    state_key = "observation.state"
    for required_key in image_keys + [state_key]:
        if required_key not in dataset.features:
            raise KeyError(
                f"数据集缺少策略需要的输入键: {required_key}。可用键: {list(dataset.features.keys())}"
            )

    video_paths = [dataset.root / dataset.meta.get_video_file_path(args.episode_index, k) for k in image_keys]
    print("=" * 80)
    print(f"Model:       {model_dir}")
    print(f"Policy:      {policy_cfg.type}")
    print(f"Device:      {device}")
    print(f"Dataset:     {repo_id}")
    print(f"DatasetRoot: {dataset.root}")
    print(f"Episode:     {args.episode_index}")
    print(f"Frames(ep):  {len(dataset)}")
    for vp in video_paths:
        print(f"Video file:  {vp}")
    print("=" * 80)

    if hasattr(policy, "reset"):
        policy.reset()

    n_steps = min(max(len(dataset), 0), max(args.max_frames, 0))
    all_actions: list[np.ndarray] = []

    compare_writer = None
    compare_video_out = None
    export_compare = bool(args.export_compare_video)
    action_labels = dataset.features.get("action", {}).get("names", [])

    for i in range(n_steps):
        item = dataset[i]

        # 组装 observation：图片(HWC,uint8) + state(float32)
        observation: dict[str, np.ndarray] = {}
        for k in image_keys:
            observation[k] = _to_hwc_uint8(item[k])

        state = _tensor_or_array_to_numpy(item[state_key]).astype(np.float32)
        observation[state_key] = state

        action_t = predict_action(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(policy_cfg.use_amp),
            task=item.get("task", None),
            robot_type=dataset.meta.robot_type,
        )

        action_np = action_t.detach().cpu().numpy().reshape(-1)
        all_actions.append(action_np)

        frame_idx = int(_tensor_or_array_to_numpy(item["frame_index"]).reshape(-1)[0])
        if i < 100 or i == n_steps - 1:
            print(f"step={i:04d} frame_index={frame_idx:04d} action={np.round(action_np, 4).tolist()}")

        if export_compare:
            try:
                import cv2
            except Exception as e:
                raise ImportError("导出对比视频需要 opencv-python（cv2）") from e

            # 取第一路视觉输入作为左侧原视频
            left_rgb = observation[image_keys[0]]
            if left_rgb.ndim != 3 or left_rgb.shape[-1] not in (3, 4):
                raise ValueError(f"无法导出对比视频，图像形状异常: {left_rgb.shape}")
            left_rgb = left_rgb[:, :, :3]

            # 初始化 writer
            if compare_writer is None:
                compare_video_out = Path(args.export_compare_video).expanduser().resolve()
                compare_video_out.parent.mkdir(parents=True, exist_ok=True)
                fps = float(args.compare_fps) if float(args.compare_fps) > 0 else float(dataset.fps)
                panel_w = max(320, int(args.compare_panel_width))
                out_h = int(left_rgb.shape[0])
                out_w = int(left_rgb.shape[1] + panel_w)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                compare_writer = cv2.VideoWriter(str(compare_video_out), fourcc, fps, (out_w, out_h))
                if not compare_writer.isOpened():
                    raise RuntimeError(f"无法创建视频写入器: {compare_video_out}")

            actions_np = np.stack(all_actions, axis=0)
            panel_rgb = _render_action_panel(
                actions=actions_np,
                height=left_rgb.shape[0],
                width=int(args.compare_panel_width),
                current_step=i,
                action_labels=list(action_labels) if isinstance(action_labels, list) else [],
            )

            if panel_rgb.shape[0] != left_rgb.shape[0]:
                panel_rgb = cv2.resize(
                    panel_rgb,
                    (int(args.compare_panel_width), int(left_rgb.shape[0])),
                    interpolation=cv2.INTER_AREA,
                )

            left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
            canvas_bgr = np.concatenate([left_bgr, right_bgr], axis=1)
            compare_writer.write(canvas_bgr)

    if len(all_actions) == 0:
        print("没有可推理的帧（请检查 episode-index 或 max-frames）")
        if compare_writer is not None:
            compare_writer.release()
        return

    actions = np.stack(all_actions, axis=0)
    print("-" * 80)
    print(f"Action shape: {actions.shape}")
    print(f"Action mean : {np.round(actions.mean(axis=0), 6).tolist()}")
    print(f"Action std  : {np.round(actions.std(axis=0), 6).tolist()}")

    if args.save_actions:
        out = Path(args.save_actions).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, actions)
        print(f"Saved actions to: {out}")

    if compare_writer is not None:
        compare_writer.release()
        print(f"Saved compare video to: {compare_video_out}")


if __name__ == "__main__":
    main()

# python /home/woan/lerobot/simulate_episode_video_inference.py --max-frames 290 --episode-index 0 --model-path /home/woan/lerobot/output_lerobot_train/act/checkpoints/030000 --export-compare-video /mnt/output_lerobot_train/act/compare_ep0_full.mp4

# python /home/woan/lerobot/simulate_episode_video_inference.py --max-frames 290 --episode-index 0 --model-path /home/woan/lerobot/output_lerobot_train/amolvla_A/checkpoints/020000 --export-compare-video /mnt/output_lerobot_train/smolvla/compare_ep0_full.mp4

# python /mnt/workspace/lerobot-xuanzi/simulate_episode_video_inference.py --max-frames 290 --episode-index 10 --model-path /mnt/workspace/lerobot-xuanzi/output_lerobot_train/smolvla_A/checkpoints/020000 --export-compare-video /mnt/output_lerobot_train/smolvla/compare_ep0_full.mp4
