import cv2
import numpy as np
import mss
import serial as pyserial
import time
import serial.tools.list_ports

monitor = {"top": 37, "left": 1161, "width": 351, "height": 758}
cup_template = cv2.imread("cup.png", 0)
waiting_template = cv2.imread("waiting.png", 0)
waiting_threshold = 0.6

match_threshold = 0.65
expected_cups_count = 10

ports = serial.tools.list_ports.comports()
ser = None
for p in ports:
    if "usbmodem" in p.device or "usbserial" in p.device:
        ser = pyserial.Serial(p.device, 9600, timeout=1)
        break
if ser is None:
    raise RuntimeError("Arduino not found. Is it connected via USB?")
time.sleep(2)

# Define known cup positions
known_cups = [
    (304, 581), (258, 531), (344, 528), (217, 480), (302,480), (382, 481), (181, 446), (427, 446),
    (302, 555), (259, 505), (344, 504), (222, 459), (382, 460),
    (302, 529), (250, 482), (331, 480)
]

# Define moves for each cup (example moves: you must fill in properly)
cup_moves = {
    0: (0, 5000), 
    1: (0, 1000), 
    2: (0, 1000), 
    3: (0, 1000), 
    4: (0, 1000), 
    5: (0, 1000), 
    6: (0, 1000), 
    7: (0, 1000), 
    8: (0, 1000), 
    9: (0, 1000), 
    10: (0, 1000), 
    11: (0, 1000), 
    12: (0, 1000), 
    13: (0, 1000), 
    14: (0, 1000),
    15: (0, 1000),
    
    # ...
}

last_coords = None
stable_count = 0
required_stability = 10  # ~10 seconds at 6 FPS

def is_stable(new_coords, last_coords, threshold=5):
    if last_coords is None:
        return False
    dx = abs(new_coords[0] - last_coords[0])
    dy = abs(new_coords[1] - last_coords[1])
    return dx < threshold and dy < threshold


def send_moves(moves):
    cmd = ""
    for m in moves:
        if isinstance(m, str):
            if m == "PENUP":
                cmd += "0,0,1,1;"     # penCommand=1, penUp=1
            elif m == "PENDOWN":
                cmd += "0,0,1,0;"     # penCommand=1, penUp=0
        else:
            dx, dy = m
            cmd += f"{dx},{dy},0,0;"  # penCommand=0, penUp ignored
    cmd += "\n"
    ser.write(cmd.encode())
    print("Sent to Arduino:", cmd.strip())


def stroke(dx, dy):
    # always lower the pen first
    send_moves(["PENDOWN"])
    ser.readline()
    time.sleep(0.2)

    # draw out
    send_moves([(dx, dy)])
    ser.readline()   # wait for “done”

    # lift the pen
    send_moves(["PENUP"])
    ser.readline()
    time.sleep(0.6)

    time.sleep(1.0)

    # return home
    send_moves([(-dx, -dy)])
    ser.readline()
    time.sleep(0.5)

    ser.flushInput()

def match_to_cup(x, y, tolerance=15):
    best_idx = -1
    best_dist2 = tolerance * tolerance
    for idx, (cx, cy) in enumerate(known_cups):
        dx = x - cx
        dy = y - cy
        dist2 = dx*dx + dy*dy
        if dist2 <= best_dist2:
            best_dist2 = dist2
            best_idx = idx
    return best_idx

def is_waiting_screen(frame_gray):
    res = cv2.matchTemplate(frame_gray, waiting_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= waiting_threshold)
    return len(loc[0]) > 0

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
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if resized_template.shape[0] > frame_gray.shape[0] or resized_template.shape[1] > frame_gray.shape[1]:
            continue
        res = cv2.matchTemplate(frame_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            rects.append([pt[0], pt[1], pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0]])
    rects = non_max_suppression_fast(rects, 0.3)
    detected_positions = [(rect[0], rect[1]) for rect in rects]
    return rects, detected_positions

with mss.mss() as sct:
    while True:
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_waiting_screen(frame_gray):
            print("Waiting for opponent... Skipping detection.")
            cv2.putText(frame, "Waiting for Opponent", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Cup Pong Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        rects, cup_positions = multi_scale_detect_cups(frame_gray, cup_template, match_threshold)

        if len(cup_positions) > 0:
            target = max(cup_positions, key=lambda p: (p[1], -p[0]))
            if is_stable(target, last_coords):
                stable_count += 1
            else:
                stable_count = 0
            last_coords = target

            if stable_count > 0:
                cv2.circle(frame, target, 10 + 2 * stable_count, (255, 0, 255), 2)
                cv2.putText(frame, f"Stable: {stable_count}", (target[0] + 15, target[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)



        if stable_count >= required_stability:
            idx = match_to_cup(*target)
            if idx != -1 and idx in cup_moves:
                dx, dy = cup_moves[idx]
                stroke(dx, dy)            # ← use helper
                known_cups.pop(idx)
                cup_moves.pop(idx)
            else:
                print("No move defined for cup", idx)
            stable_count = 0



        for rect in rects:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        if len(cup_positions) == expected_cups_count:
            cv2.putText(frame, "Game Start Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print("Game start detected. Cup positions (x, y):", cup_positions)
        else:
            print("Current cup detections:", cup_positions)

        cv2.imshow("Cup Pong Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()