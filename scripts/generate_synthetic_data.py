"""
生成仿真眼动数据，模拟GazeBase格式

用于测试pipeline，无需下载真实数据。

Usage:
    python scripts/generate_synthetic_data.py --output_dir data/synthetic_gazebase --num_subjects 50
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_subject_gaze_pattern(subject_id, task, duration_samples=5000, sampling_rate=1000):
    """
    为每个被试生成独特的眼动模式（带个体差异）

    模拟真实眼动特征：
    - 个体特异的扫视速度偏好
    - 个体特异的注视分散度
    - 任务相关的通用模式
    """
    np.random.seed(subject_id * 1000 + hash(task) % 1000)

    # 个体参数（生物特征，用于身份识别）
    subject_saccade_velocity_bias = np.random.uniform(80, 150)  # deg/s
    subject_fixation_dispersion = np.random.uniform(0.5, 2.0)   # deg
    subject_pupil_baseline = np.random.uniform(3.0, 5.0)        # mm

    # 任务参数（任务通用模式）
    if task == "FXS":  # Fixation
        pattern_type = "fixation"
        movement_scale = 0.3
    elif task == "HSS":  # Horizontal saccade
        pattern_type = "horizontal_saccade"
        movement_scale = 1.0
    elif task == "RAN":  # Random saccade
        pattern_type = "random_saccade"
        movement_scale = 1.5
    elif task == "TEX":  # Reading
        pattern_type = "reading"
        movement_scale = 0.8
    else:  # VD1/VD2
        pattern_type = "video"
        movement_scale = 1.2

    # 生成数据
    data = []
    t = 0
    x_pos = 0.0
    y_pos = 0.0

    while t < duration_samples:
        # 决定事件类型（扫视 vs 注视）
        if np.random.rand() < 0.3:  # 30% 扫视
            # 扫视事件
            saccade_duration = int(np.random.uniform(20, 60))  # 20-60ms
            target_x = np.random.uniform(-10, 10) * movement_scale
            target_y = np.random.uniform(-8, 8) * movement_scale

            if pattern_type == "horizontal_saccade":
                target_y = 0  # 只水平移动
            elif pattern_type == "reading":
                target_x = np.random.uniform(0, 5)  # 向右阅读
                target_y = np.random.uniform(-1, 1)

            # 生成扫视轨迹（主序列关系：幅度越大速度越快）
            saccade_amplitude = np.sqrt((target_x - x_pos)**2 + (target_y - y_pos)**2)
            saccade_velocity = subject_saccade_velocity_bias + saccade_amplitude * 10

            for i in range(saccade_duration):
                alpha = i / saccade_duration
                x_curr = x_pos + (target_x - x_pos) * alpha
                y_curr = y_pos + (target_y - y_pos) * alpha

                pupil = subject_pupil_baseline + np.random.normal(0, 0.1)

                data.append({
                    'n': t,
                    'x': x_curr,
                    'y': y_curr,
                    'pupil': pupil,
                    'valid': 1
                })
                t += 1

            x_pos, y_pos = target_x, target_y

        else:  # 70% 注视
            # 注视事件
            fixation_duration = int(np.random.uniform(150, 500))  # 150-500ms

            for i in range(fixation_duration):
                # 注视漂移（微小抖动）
                x_curr = x_pos + np.random.normal(0, subject_fixation_dispersion * 0.1)
                y_curr = y_pos + np.random.normal(0, subject_fixation_dispersion * 0.1)

                # 瞳孔直径（认知负荷相关）
                pupil = subject_pupil_baseline + np.random.normal(0, 0.2)

                # 偶尔的眨眼（丢失数据）
                valid = 0 if np.random.rand() < 0.01 else 1

                data.append({
                    'n': t,
                    'x': x_curr,
                    'y': y_curr,
                    'pupil': pupil if valid else np.nan,
                    'valid': valid
                })
                t += 1

    return pd.DataFrame(data[:duration_samples])


def generate_synthetic_gazebase(output_dir, num_subjects=50, num_rounds=2, tasks=None):
    """
    生成仿真GazeBase数据集

    Args:
        output_dir: 输出目录
        num_subjects: 被试数量（真实GazeBase有322人）
        num_rounds: 会话轮次（真实有9轮）
        tasks: 任务列表
    """
    output_path = Path(output_dir)
    tasks = tasks or ["FXS", "HSS", "RAN", "TEX"]

    print(f"Generating synthetic GazeBase dataset:")
    print(f"  Output: {output_path}")
    print(f"  Subjects: {num_subjects}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Tasks: {tasks}")
    print()

    total_files = num_subjects * num_rounds * len(tasks)

    with tqdm(total=total_files, desc="Generating") as pbar:
        for round_id in range(1, num_rounds + 1):
            round_dir = output_path / f"Round_{round_id}"
            round_dir.mkdir(parents=True, exist_ok=True)

            for subject_id in range(1001, 1001 + num_subjects):
                for task in tasks:
                    # 文件名格式: S_{subject_id}_S{session}_{task}.csv
                    filename = f"S_{subject_id}_S{round_id}_{task}.csv"
                    filepath = round_dir / filename

                    # 生成数据
                    df = generate_subject_gaze_pattern(
                        subject_id=subject_id,
                        task=task,
                        duration_samples=5000,  # 5秒数据
                        sampling_rate=1000
                    )

                    # 保存CSV
                    df.to_csv(filepath, index=False)
                    pbar.update(1)

    # 生成统计信息
    print("\n" + "="*60)
    print("Dataset generated successfully!")
    print("="*60)
    print(f"Total files: {total_files}")
    print(f"Total size: ~{total_files * 0.2:.1f} MB (estimated)")
    print("\nDirectory structure:")
    for round_id in range(1, num_rounds + 1):
        round_dir = output_path / f"Round_{round_id}"
        n_files = len(list(round_dir.glob("*.csv")))
        print(f"  Round_{round_id}/: {n_files} files")
    print("\nYou can now run:")
    print(f"  python scripts/train.py --config configs/default.yaml")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic eye-tracking data")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_gazebase")
    parser.add_argument("--num_subjects", type=int, default=50, help="Number of subjects (default: 50)")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of rounds (default: 2)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["FXS", "HSS", "RAN", "TEX"])
    args = parser.parse_args()

    generate_synthetic_gazebase(
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        num_rounds=args.num_rounds,
        tasks=args.tasks
    )


if __name__ == "__main__":
    main()
