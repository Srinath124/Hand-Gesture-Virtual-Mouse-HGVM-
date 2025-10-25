import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
# Parameters
##########################
wCam, hCam = 640, 480
frameR = 100  # Frame reduction
smoothening = 7  # Lower = more responsive, Higher = smoother

##########################
# Variables
##########################
pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

# Click debounce to prevent multiple clicks
click_cooldown = 0
right_click_cooldown = 0
COOLDOWN_TIME = 0.3  # 300ms between clicks

##########################
# Camera Setup
##########################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

##########################
# Hand Detector
##########################
detector = htm.handDetector(maxHands=1, detectionCon=0.7)
wScr, hScr = autopy.screen.size()
print(f"Screen size: {wScr} x {hScr}")

##########################
# Main Loop
##########################
while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    # 2. Get the tip of the index, middle, and ring fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger
        x3, y3 = lmList[16][1:]  # Ring finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        
        # Draw frame boundary
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                     (255, 0, 255), 2)

        # 4. Only Index Finger Up: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3_screen = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3_screen = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen Values
            cLocX = pLocX + (x3_screen - pLocX) / smoothening
            cLocY = pLocY + (y3_screen - pLocY) / smoothening

            # 7. Move Mouse
            try:
                autopy.mouse.move(wScr - cLocX, cLocY)
            except Exception as e:
                print(f"Mouse move error: {e}")
                
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, "MOVE", (x1 + 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 0, 255), 2)
            pLocX, pLocY = cLocX, cLocY

        # 8. Index and Middle Fingers Up: Left Click Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # 10. Click if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                current_time = time.time()
                if current_time - click_cooldown > COOLDOWN_TIME:
                    try:
                        autopy.mouse.click()
                        click_cooldown = current_time
                        cv2.putText(img, "LEFT CLICK", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 255, 0), 3)
                    except Exception as e:
                        print(f"Click error: {e}")
            else:
                cv2.putText(img, "READY TO CLICK", (lineInfo[4] - 80, lineInfo[5] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 11. Index, Middle, and Ring Fingers Up: Right Click Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            # 12. Find distance between middle and ring fingers
            length, img, lineInfo = detector.findDistance(12, 16, img)

            # 13. Right click if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 0, 255), cv2.FILLED)
                current_time = time.time()
                if current_time - right_click_cooldown > COOLDOWN_TIME:
                    try:
                        autopy.mouse.click(autopy.mouse.Button.RIGHT)
                        right_click_cooldown = current_time
                        cv2.putText(img, "RIGHT CLICK", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 0, 255), 3)
                    except Exception as e:
                        print(f"Right click error: {e}")
            else:
                cv2.putText(img, "READY TO R-CLICK", (lineInfo[4] - 80, lineInfo[5] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Display finger status
        finger_status = f"Fingers: {fingers}"
        cv2.putText(img, finger_status, (20, hCam - 30), cv2.FONT_HERSHEY_PLAIN, 
                   1, (255, 255, 255), 2)

    # 14. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 
               3, (255, 0, 0), 3)

    # Instructions
    cv2.putText(img, "1 finger: Move | 2 fingers: L-Click | 3 fingers: R-Click", 
               (20, hCam - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Display
    cv2.imshow("Hand Gesture Mouse Control", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
