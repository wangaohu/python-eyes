"""
python-eyes —— 基于 OpenCV 的实时颜色识别与计数 Demo

用法：
    python main.py                # 使用默认摄像头
    python main.py --camera 1     # 换一个摄像头
    python main.py --no-window    # 不弹窗口，只写日志（用于无显示环境）

运行后按 q 退出。每帧识别到的色块会追加写入 log.csv。
"""

import argparse
import csv
import sys
import time
from collections import Counter

import cv2
import numpy as np

# ---------------------------------------------------------------- 可调参数
AREA_MIN = 800          # 面积阈值：小于这个像素面积的色块当作噪点丢掉
FRAME_W, FRAME_H = 640, 480
LOG_PATH = "log.csv"

# 颜色名 -> HSV 双阈值。OpenCV 的 H 范围是 0-179
COLOR_RANGES = {
    "red":    ((0, 120, 70), (10, 255, 255)),
    "green":  ((40, 70, 70), (85, 255, 255)),
    "blue":   ((95, 120, 70), (135, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
}


def open_camera(index: int):
    """打开摄像头。Windows 上优先用 DirectShow，打开更快也更稳。"""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            return cap
        cap.release()
    return None


def find_blobs(frame, name, lo, hi):
    """在画面里找出某一种颜色的色块，返回 [(x, y, w, h, area), ...]。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    # 先腐蚀再膨胀：消掉零星噪点，同时保住主体形状
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < AREA_MIN:          # 面积过滤，这是把误检压下去的关键一步
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blobs.append((x, y, w, h, area))
    return blobs


def main() -> int:
    parser = argparse.ArgumentParser(description="实时颜色识别与计数 Demo")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--no-window", action="store_true", help="不显示预览窗口")
    args = parser.parse_args()

    cap = open_camera(args.camera)
    if cap is None:
        print(f"[错误] 打不开摄像头 {args.camera}。", file=sys.stderr)
        print("       笔记本内置摄像头通常是 0，外接的试试 1。", file=sys.stderr)
        print("       或者检查：设置 → 隐私和安全性 → 相机 → 允许应用访问相机。", file=sys.stderr)
        return 1

    log_file = open(LOG_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "color", "x", "y", "width", "height", "area"])

    counters = Counter()
    frames = 0
    fps = 0.0
    prev_time = time.time()
    failed = 0

    print(f"摄像头 {args.camera} 已打开，窗口里按 q 退出……")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed += 1
                if failed > 30:
                    print("[错误] 连续读取失败，摄像头可能被其他程序占用了。", file=sys.stderr)
                    break
                continue
            failed = 0
            frames += 1
            now = time.time()

            for name, (lo, hi) in COLOR_RANGES.items():
                for (x, y, w, h, area) in find_blobs(frame, name, lo, hi):
                    counters[name] += 1
                    writer.writerow([round(now, 2), name, x, y, w, h, int(area)])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # 注意：OpenCV 画不了中文，屏幕上的字只能用英文
                    cv2.putText(frame, f"{name} {int(area)}", (x, max(y - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS 做一点平滑，不然数字会乱跳
            instant = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
            cv2.putText(frame, f"FPS {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            if not args.no_window:
                cv2.imshow("python-eyes", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frames % 10 == 0:
                log_file.flush()
    except KeyboardInterrupt:
        print("\n手动中断。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()

    print(f"\n处理帧数：{frames}，平均 FPS：{fps:.1f}")
    if counters:
        for name, count in counters.most_common():
            print(f"  {name}: 检测到 {count} 次")
    else:
        print("  一帧都没识别到色块，检查一下画面里有没有红/绿/蓝/黄的物体。")
    print(f"明细已写入 {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
