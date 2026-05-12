import argparse
import time

import cv2
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Post local webcam frames to the germination monitor.")
    parser.add_argument("--url", default="http://localhost:5000/upload_image", help="Flask upload endpoint.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between captures.")
    parser.add_argument("--seed-count", type=int, default=25, help="Expected number of seeds in the tray.")
    return parser.parse_args()


def main():
    args = parse_args()
    capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        raise SystemExit(f"Unable to open webcam index {args.camera}.")

    print(f"Posting webcam frames to {args.url}?seed_count={args.seed_count}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Frame capture failed.")
                time.sleep(args.interval)
                continue

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                print("JPEG encoding failed.")
                time.sleep(args.interval)
                continue

            response = requests.post(
                args.url,
                params={"seed_count": args.seed_count},
                data=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            print(
                f"{data['timestamp']} | sprouts={data['sprout_count']} "
                f"area={data['total_white_area']} germination={data['germination_percentage']}%"
            )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
