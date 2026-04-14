import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

BG_BGR = (0, 255, 0)
MODEL_NAME = "selfie_multiclass_256x256.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)


def ensure_model():
    p = Path(__file__).resolve().parent / MODEL_NAME
    if not p.exists():
        subprocess.run(["curl", "-fsSL", MODEL_URL, "-o", str(p)])
    return str(p)


def mask_from_result(res, h, w):
    masks = res.confidence_masks or []
    if len(masks) >= 6:
        bg = np.asarray(masks[0].numpy_view(), dtype=np.float32)
        person = 1.0 - np.clip(bg, 0.0, 1.0)
    elif len(masks) >= 2:
        person = np.asarray(masks[1].numpy_view(), dtype=np.float32)
    elif len(masks) == 1:
        person = np.asarray(masks[0].numpy_view(), dtype=np.float32)
    elif res.category_mask is not None:
        cat = np.squeeze(np.asarray(res.category_mask.numpy_view()))
        person = (cat > 0).astype(np.float32)
    else:
        raise SystemExit("сегментатор не вернул маски")
    person = np.squeeze(person)
    if person.shape[0] != h or person.shape[1] != w:
        person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
    return person

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("video", nargs="?", default="")
    ap.add_argument("-c", "--camera", type=int, default=None)
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise SystemExit(f"Камера {args.camera} не открывается")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = Path(args.output) if args.output else None
        use_cam = True
    else:
        if not args.video:
            raise SystemExit("Укажите путь к видео или --camera")
        inp = Path(args.video)
        if not inp.is_file():
            raise SystemExit(f"Нет файла: {inp}")
        cap = cv2.VideoCapture(str(inp))
        if not cap.isOpened():
            raise SystemExit(f"Не открывается: {inp}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out = Path(args.output) if args.output else inp.with_name(inp.stem + "_nobg.mp4")
        use_cam = False

    writer = None
    times = []
    n = 0
    with ImageSegmenter.create_from_options(opts) as seg:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            h, w = frame.shape[:2]

            if writer is None and out is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"не удалось создать выход: {out}")

            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(round((n - 1) * 1000.0 / max(fps, 1e-6)))
            res = seg.segment_for_video(mp_image, ts_ms)
            person = mask_from_result(res, h, w)

            m3 = np.repeat(np.clip(person, 0.0, 1.0)[:, :, None], 3, axis=2)
            bg = np.empty_like(frame)
            bg[:, :] = BG_BGR
            out_f = (frame.astype(np.float32) * m3 + bg.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if len(times) > 120:
                times.pop(0)

            if writer is not None:
                writer.write(out_f)

            if use_cam:
                avg = sum(times) / len(times)
                fps_proc = 1.0 / avg if avg > 1e-9 else 0.0
                label = f"{fps_proc:.1f} FPS  {w}x{h}"
                cv2.putText(out_f, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out_f, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("camera", out_f)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            elif n % 30 == 0 and times:
                dt = sum(times[-30:]) / min(30, len(times))
                print(f"кадр {n}  ~{1.0/dt:.1f} FPS")

    cap.release()
    if writer is not None:
        writer.release()
    if use_cam:
        cv2.destroyAllWindows()

    if times and not use_cam:
        avg = sum(times) / len(times)
        print(f"Видео {out} обработано:\nКадров: {n} | среднее {1000*avg:.1f} ms/кадр  | {1.0/avg:.1f} FPS")
    elif times and use_cam and out:
        avg = sum(times) / len(times)
        print(f"Видео {out}записано. Кадров: {n}  ~{1.0/avg:.1f} FPS")


if __name__ == "__main__":
    main()
