import cv2
import mediapipe as mp

hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Detect hands
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Draw if hands found
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            draw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
