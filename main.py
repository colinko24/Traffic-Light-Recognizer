import cv2
import os
from traffic_engine import TrafficSystem

def run_app(mode='live', input_path=None):
    system = TrafficSystem()
    
    # Determine Source
    source = 0 if mode == 'live' else input_path
    cap = cv2.VideoCapture(source)
    
    # Setup Video Writer if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = None

    print(f"Starting {mode} mode. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame using the optimized engine
        annotated = system.process_frame(frame)

        # Lazy-initialize VideoWriter based on frame size
        if out_writer is None:
            h, w, _ = annotated.shape
            out_writer = cv2.VideoWriter('output_annotated.mp4', fourcc, 20.0, (w, h))

        out_writer.write(annotated)
        cv2.imshow('Traffic Recognition', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # OPTIONS: 'live' for webcam, 'video' for file
    run_app(mode='video', input_path='NEWYORK.mp4')
