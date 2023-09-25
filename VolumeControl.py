import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Set the camera width and height
wCam, hCam = 640, 480

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Set the camera width and height
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize the previous time variable for FPS calculation
pTime = 0

# Initialize the HandTrackingModule detector
detector = htm.handDetector(detectionCon=0.7)

# Get the audio devices and set up audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get the volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Initialize variables for volume control
vol = 0
volB = 400
volP = 0
fixVolume = False

# Initialize the initial distance between thumb and forefinger as None
initial_distance = None

# Initialize a flag for middle finger up detection
middle_finger_up = False

# Initialize a flag for "thumbs up" gesture detection
thumbs_up = False

while True:
    success, img = cap.read()

    # Flip the image horizontally for a more intuitive user experience
    img = cv2.flip(img, 1)

    # Check if the frame was successfully read
    if not success:
        print("Error: Failed to read frame.")
        break

    # Find hands in the frame
    img = detector.findHands(img)
    lmList = detector.findPos(img, draw=False)

    # Check if hand landmarks were detected
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        x_middle, y_middle = lmList[12][1], lmList[12][2]

        # Calculate the real-world distance between thumb and forefinger only once
        if initial_distance is None:
            initial_distance = math.hypot(x2 - x1, y2 - y1)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw landmarks and lines on the hand
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Normalize the hand gesture's length
        normalized_length = length / initial_distance

        # Check if the middle finger is up
        if y_middle < 200:
            middle_finger_up = True
        else:
            middle_finger_up = False

        # Toggle the fixVolume flag when the middle finger is up
        if middle_finger_up:
            fixVolume = not fixVolume

        # Display volume lock status on the screen
        if fixVolume:
            cv2.putText(
                img,
                "Volume Fixed",
                (310, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),  # Red color
                3,
            )
        else:
            cv2.putText(
                img,
                "Volume Free",
                (310, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),  # Green color
                3,
            )

        if not fixVolume:
            # Adjust the scaling factor for volume control
            vol = np.interp(normalized_length, [0.1, 1], [minVol, maxVol])
            vol = max(
                0.0, min(1.0, (vol - minVol) / (maxVol - minVol))
            )  # Ensure vol is within the valid range
            volB = np.interp(normalized_length, [0.1, 1], [400, 150])
            volP = np.interp(normalized_length, [0.1, 1], [0, 100])

            # Set the master volume level scalar for more accurate control
            volume.SetMasterVolumeLevelScalar(vol, None)

        # Check for "thumbs up" gesture
        if (
            lmList[4][2] < lmList[8][2]  # Thumb tip is above thumb base
            and lmList[8][2]
            < lmList[12][2]  # Index finger tip is above index finger base
            and lmList[12][2]
            < lmList[16][2]  # Middle finger tip is above middle finger base
            and lmList[16][2]
            < lmList[20][2]  # Ring finger tip is above ring finger base
        ):
            thumbs_up = True
        else:
            thumbs_up = False

        # Close the window if a "thumbs up" gesture is detected
        if thumbs_up:
            cv2.destroyAllWindows()
            break

    # Draw volume control elements on the screen
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volB)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(
        img, f"{int(volP)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3
    )

    # Calculate and display the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3
    )

    # Display the image
    cv2.imshow("Volume Gesture Conroller", img)

    # Check for the Esc key to exit the loop
    if cv2.waitKey(1) == 27:  # Esc key
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
