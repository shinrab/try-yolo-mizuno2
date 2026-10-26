import argparse
import platform
import sys

import cv2
from ultralytics import YOLO


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera with the native backend for this operating system."""
    system = platform.system()
    backend_names = {
        "Darwin": ("AVFoundation", cv2.CAP_AVFOUNDATION),
        "Windows": ("Media Foundation", cv2.CAP_MSMF),
        "Linux": ("Video4Linux2", cv2.CAP_V4L2),
    }
    attempts = []
    native = backend_names.get(system)
    if native:
        attempts.append(native)
    # macOSのCAP_ANYもAVFoundationへ進むため、同じアクセス拒否を
    # 繰り返さない。ほかのOSでは自動バックエンドも試す。
    if system != "Darwin":
        attempts.append(("OpenCV automatic backend", cv2.CAP_ANY))

    for name, backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"カメラ {index} を {name} で開きました。")
            return cap
        cap.release()
    raise RuntimeError(
        f"カメラ {index} を開けませんでした。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOで人物をリアルタイム追跡します")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="使用するカメラ番号 (既定: 0)",
    )
    args = parser.parse_args()

    cap = None
    try:
        # 先にカメラを開き、失敗した場合にモデルを読み込まない。
        cap = open_camera(args.camera)
        model = YOLO("yolo11n.pt")
        print("リアルタイム認識を開始します。終了するには 'q' を押してください。")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("エラー: カメラからフレームを取得できません。")
                break

            results = model.track(
                source=frame, persist=True, classes=[0], verbose=False
            )
            cv2.imshow("YOLO Real-time Human Tracking", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except RuntimeError as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
