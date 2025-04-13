import cv2
import numpy as np
import mss

monitor = {"top": 77, "left": 1208, "width": 295, "height": 643}

cup_template = cv2.imread("cup.png", 0)

match_threshold = 0.7
expected_cups_count = 10


def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [len(idxs) - 1]
        for pos in range(len(idxs) - 1):
            i = idxs[pos]
            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[i]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return boxes[pick].astype("int").tolist()


def multi_scale_detect_cups(frame_gray, template, threshold, scales=np.linspace(0.8, 1.2, 5)):
    rects = []
    for scale in scales:
        # Resize template for the current scale
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Skip if the resized template is larger than the frame
        if resized_template.shape[0] > frame_gray.shape[0] or resized_template.shape[1] > frame_gray.shape[1]:
            continue
        res = cv2.matchTemplate(frame_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            rects.append([pt[0], pt[1], pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0]])
    # Apply non-maximum suppression to reduce overlapping detections
    rects = non_max_suppression_fast(rects, 0.3)
    # Extract top-left positions from the rectangles
    detected_positions = [(rect[0], rect[1]) for rect in rects]
    return rects, detected_positions


with mss.mss() as sct:
    while True:
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        # Convert frame to grayscale for matching
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use multi-scale template matching to detect cups
        rects, cup_positions = multi_scale_detect_cups(frame_gray, cup_template, match_threshold)

        # Draw rectangles around each detected cup
        for rect in rects:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        # Check if the game is in the 'start' position (based on expected cup count)
        if len(cup_positions) == expected_cups_count:
            cv2.putText(frame, "Game Start Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print("Game start detected. Cup positions (x, y):", cup_positions)
        else:
            print("Current cup detections:", cup_positions)

        # Display the processed frame
        cv2.imshow("Cup Pong Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()