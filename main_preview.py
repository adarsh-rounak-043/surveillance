# main_preview.py
import cv2
import time
import argparse

from captioner import Blip2Captioner
from caption_classifier import CaptionClassifier
from event_logic import decide_event_action


def put_multiline_text(
    img,
    text: str,
    org=(10, 25),
    line_height: int = 20,
    color=(0, 255, 0),
    thickness: int = 1,
):
    """
    Draw simple multi-line text on the frame.
    """
    x, y = org
    for line in text.split("\n"):
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def run(camera_source=0):
    # 1. Initialize models
    print("Initializing models...")
    captioner = Blip2Captioner()
    classifier = CaptionClassifier()

    # 2. Open camera (0 = default webcam; string = RTSP/HTTP URL)
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: could not open camera source: {camera_source}")
        return

    FRAME_INTERVAL = 10  # run caption+classifier every 10th frame
    frame_idx = 0

    last_caption = ""
    last_label = ""
    last_conf = 0.0
    last_time = 0.0

    print("Starting real-time preview. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, retrying...")
            time.sleep(0.5)
            continue

        frame_idx += 1

        # Run heavy models only every FRAME_INTERVAL frames
        if frame_idx % FRAME_INTERVAL == 0:
            timestamp = time.time()
            try:
                # 3. Generate caption
                caption = captioner.generate_caption(frame)
                # 4. Classify caption
                label, confidence = classifier.predict(caption)
                # 5. Decide if alert
                action = decide_event_action(label, confidence)

                if action.raise_alert:
                    print(f"[ALERT] {label} ({confidence:.2f}) @ {timestamp}")
                    print(f"  Caption: {caption}")

                last_caption = caption
                last_label = label
                last_conf = confidence
                last_time = timestamp

            except Exception as e:
                print(f"Error in processing frame: {e}")

        # 6. Draw overlay using last caption/label
        overlay_lines = []
        if last_caption:
            overlay_lines.append(f"Caption: {last_caption[:80]}")  # truncate
        if last_label:
            overlay_lines.append(f"Event: {last_label} ({last_conf:.2f})")
        if last_time:
            overlay_lines.append(time.strftime("Time: %H:%M:%S",
                                               time.localtime(last_time)))

        if overlay_lines:
            put_multiline_text(
                frame,
                "\n".join(overlay_lines),
                org=(10, 25),
                color=(0, 255, 0),
                thickness=1,
            )

        cv2.imshow("Campus Surveillance Preview", frame)

        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera index (e.g. 0) or RTSP/HTTP URL",
    )
    args = parser.parse_args()

    # Try to parse as integer webcam index; else treat as URL
    try:
        cam_source = int(args.camera)
    except ValueError:
        cam_source = args.camera

    run(cam_source)
