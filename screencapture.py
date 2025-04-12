import cv2
import numpy as np
import mss

# Define the region of the screen to capture.
monitor = {"top": 77, "left": 1208, "width": 295, "height": 643}

with mss.mss() as sct:
    while True:
        # Grab a screenshot of the defined monitor region.
        sct_img = sct.grab(monitor)
        # Convert the raw data to a NumPy array and then to BGR format for OpenCV.
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Re-load the template image every iteration, so changes in the file are detected.
        template = cv2.imread("template.png", 0)
        if template is not None:
            # Print the shape of the template.
            print(f"Template shape: {template.shape}")

            template_w, template_h = template.shape[::-1]
            # Convert the frame to grayscale for template matching.
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"Captured gray frame shape: {gray_frame.shape}")

            # Display the template for visual verification.
            cv2.imshow("Template", template)

            # Perform template matching using normalized correlation coefficient.
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # Print matching results to the console
            print(f"Match value: {max_val:.3f} at location: {max_loc}")

            # For debugging: also display the result map (optional)
            cv2.imshow("Match Result", res)

            # Define a threshold for a valid detection.
            threshold = 0.9  # Even with threshold 0, if max_val is negative, nothing will be detected.
            if max_val >= threshold:
                # When detected, draw a rectangle around the match location and display a label.
                top_left = max_loc
                bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                cv2.putText(frame, "GamePigeon Cup Pong", (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Detected GamePigeon Cup Pong")
            else:
                print("GamePigeon Cup Pong not detected (score below threshold)")
        else:
            # If the template file is not found, indicate on the frame and print the message.
            cv2.putText(frame, "Template not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("Template not found")

        # Display the captured frame.
        cv2.imshow("Mirrored iPhone Screen", frame)

        # Exit on 'q' key press.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()