import cv2
import time
import datetime

cap = cv2.VideoCapture(0)  # 0 is the id of the camera

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Load the cascade trained on thousands of photos

body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")  # Load the cascade trained on thousands of photos

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
    _, frame = cap.read()  # Read the frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5)  # Detect the faces in the frame
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False  # Reset the timer
        else:
            detection = True  # Start the timer
            current_time = datetime.datetime.now().strftime(
                "%d-%m-%Y-%H-%M-%S")  # Get the time when the detection stopped
            out = cv2.VideoWriter(f"{current_time}.mp4",
                                  fourcc, 20, frame_size)  # Create a video file
            print("Started recording!")
    elif detection:
        if timer_started:
            if time.time() + detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stopped recording!")
            else:
                timer_started = True  # Start the timer
                detection_stopped_time = time.time()  # Get the time when the detection stopped

    if detection:
        out.write(frame)  # Write the frame to the video file

    cv2.imshow("Camera", frame)  # Display the frame

    if cv2.waitKey(1) == ord('q'):  # Exit if the user presses 'q'
        break

out.release()  # Release the video writer
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all the frames
