import speech_recognition as sr
import pyttsx3
from threading import Thread
from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import os

# Flask app initialization
app = Flask(__name__)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video Capture setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Unable to access the camera.")

# Folder paths for clothing categories
categories = {
    "traditional": {
        "shirts": r"C:\Users\Lenovo\Desktop\Final_Project\app\Traditional\Shirts",
        "pants": r"C:\Users\Lenovo\Desktop\Final_Project\app\Traditional\Pants"
    },
    "western": {
        "shirts": r"C:\Users\Lenovo\Desktop\Final_Project\app\Western\Shirts",
        "pants": r"C:\Users\Lenovo\Desktop\Final_Project\app\Western\Pants"
    },
    "casual": {
        "shirts": r"C:\Users\Lenovo\Desktop\Final_Project\app\Casual\Shirts",
        "pants": r"C:\Users\Lenovo\Desktop\Final_Project\app\Casual\Pants"
    }
}

# Global variables
shirtImageNumber = 0
pantsImageNumber = 0
shirt_offset_x = 0
shirt_offset_y = 0
pants_offset_x = 0
pants_offset_y = 0
listShirts = []
listPants = []

fixedRatio = 320 / 200  # Shoulder width adjustment
shirtRatioHeightWidth = 650 / 510  # Shirt aspect ratio
pantsRatioHeightWidth = 1000 / 510  # Pants aspect ratio

# Function to overlay images with transparency
def overlay_image_alpha(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    h, w = overlay.shape[:2]

    # Check bounds
    if x + w > background_width or y + h > background_height:
        w = min(w, background_width - x)
        h = min(h, background_height - y)
        overlay = cv2.resize(overlay, (w, h))

    if overlay.shape[2] < 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image
    return background

# Frame generator with dynamic category
def generate_frames(category):
    global shirtImageNumber, pantsImageNumber, shirt_offset_x, shirt_offset_y, pants_offset_x, pants_offset_y, listShirts, listPants

    # Select the appropriate folder paths
    if category not in categories:
        raise ValueError(f"Category '{category}' not found.")

    shirtFolderPath = categories[category]["shirts"]
    pantsFolderPath = categories[category]["pants"]

    # Get list of shirt and pants images (filter only valid image files)
    listShirts = [f for f in os.listdir(shirtFolderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    listPants = [f for f in os.listdir(pantsFolderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Debugging: Print the list of files detected
    print(f"Shirts List: {listShirts}")
    print(f"Pants List: {listPants}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(image_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            ih, iw, _ = frame.shape

            # Landmarks for overlay positioning
            lm11 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            lm12 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lm23 = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            lm24 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            lm11_px = (int(lm11.x * iw), int(lm11.y * ih))
            lm12_px = (int(lm12.x * iw), int(lm12.y * ih))
            lm23_px = (int(lm23.x * iw), int(lm23.y * ih))
            lm24_px = (int(lm24.x * iw), int(lm24.y * ih))

            # Pants overlay
            pants_width = int(abs(lm23_px[0] - lm24_px[0]) * fixedRatio)
            pants_height = int(pants_width * pantsRatioHeightWidth)

            if pants_width > 0 and pants_height > 0:
                pants_top_left = (
                    max(0, min(iw - pants_width, min(lm23_px[0], lm24_px[0]) - int(pants_width * 0.15)) + pants_offset_x),
                    max(0, min(ih - pants_height, min(lm23_px[1], lm24_px[1])) + pants_offset_y)
                )

                imgPantsPath = os.path.join(pantsFolderPath, listPants[pantsImageNumber])
                imgPants = cv2.imread(imgPantsPath, cv2.IMREAD_UNCHANGED)
                if imgPants is not None:
                    imgPants = cv2.resize(imgPants, (pants_width, pants_height))
                    frame = overlay_image_alpha(frame, imgPants, pants_top_left[0], pants_top_left[1])

            # Shirt overlay
            shirt_width = int(abs(lm11_px[0] - lm12_px[0]) * fixedRatio)
            shirt_height = int(shirt_width * shirtRatioHeightWidth)

            if shirt_width > 0 and shirt_height > 0:
                shirt_top_left = (
                    max(0, min(iw - shirt_width, min(lm11_px[0], lm12_px[0]) - int(shirt_width * 0.15)) + shirt_offset_x),
                    max(0, min(ih - shirt_height, min(lm11_px[1], lm12_px[1]) - int(shirt_height * 0.18)) + shirt_offset_y)
                )

                imgShirtPath = os.path.join(shirtFolderPath, listShirts[shirtImageNumber])
                imgShirt = cv2.imread(imgShirtPath, cv2.IMREAD_UNCHANGED)
                if imgShirt is not None:
                    imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height))
                    frame = overlay_image_alpha(frame, imgShirt, shirt_top_left[0], shirt_top_left[1])

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

# Function to continuously listen for voice commands in always listening mode
def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    global shirtImageNumber, pantsImageNumber, shirt_offset_x, shirt_offset_y, pants_offset_x, pants_offset_y

    while True:
        try:
            with mic as source:
                print("Listening for voice command...")
                recognizer.adjust_for_ambient_noise(source)  # Adjusts for ambient noise
                audio = recognizer.listen(source, timeout=5)  # Listen for commands with a timeout
                command = recognizer.recognize_google(audio).lower()
                print(f"Voice Command: {command}")

                # Add length checks before performing modulo operations
                if len(listShirts) > 0:
                    if "next shirt" in command:
                        shirtImageNumber = (shirtImageNumber + 1) % len(listShirts)
                        engine.say("Next shirt selected")
                        engine.runAndWait()

                    elif "previous shirt" in command:
                        shirtImageNumber = (shirtImageNumber - 1) % len(listShirts)
                        engine.say("Previous shirt selected")
                        engine.runAndWait()

                if len(listPants) > 0:
                    if "next pants" in command:
                        pantsImageNumber = (pantsImageNumber + 1) % len(listPants)
                        engine.say("Next pants selected")
                        engine.runAndWait()

                    elif "previous pants" in command:
                        pantsImageNumber = (pantsImageNumber - 1) % len(listPants)
                        engine.say("Previous pants selected")
                        engine.runAndWait()

                # Offset adjustment commands remain the same
                if "shirt up" in command:
                    shirt_offset_y -= 5
                    engine.say("Shirt moved up")
                    engine.runAndWait()

                elif "shirt down" in command:
                    shirt_offset_y += 5
                    engine.say("Shirt moved down")
                    engine.runAndWait()

                elif "pants up" in command:
                    pants_offset_y -= 5
                    engine.say("Pants moved up")
                    engine.runAndWait()

                elif "pants down" in command:
                    pants_offset_y += 5
                    engine.say("Pants moved down")
                    engine.runAndWait()

        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")
        except sr.RequestError:
            print("Sorry, there was an error with the speech recognition service.")
        except Exception as e:
            print(f"Error occurred: {e}")

# Routes for HTML pages
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/trynow')
def trynow():
    return render_template('trynow.html')
    
@app.route('/traditional')
def traditional():
    return render_template('indexi.html')

@app.route('/western')
def western():
    return render_template('indexw.html')

@app.route('/casual')
def casual():
    return render_template('indexc.html')

@app.route('/update_offsets')
def update_offsets():
    key = request.args.get('key', type=str)

    global shirtImageNumber, pantsImageNumber, shirt_offset_x, shirt_offset_y, pants_offset_x, pants_offset_y

    # Modify offsets based on key press
    if key == 'w':  # Move shirt up
        shirt_offset_y -= 5
    elif key == 's':  # Move shirt down
        shirt_offset_y += 5
    elif key == 'a':  # Move shirt left
        shirt_offset_x -= 5
    elif key == 'd':  # Move shirt right
        shirt_offset_x += 5
    elif key == 'i':  # Move pants up
        pants_offset_y -= 5
    elif key == 'k':  # Move pants down
        pants_offset_y += 5
    elif key == 'j':  # Move pants left
        pants_offset_x -= 5
    elif key == 'l':  # Move pants right
        pants_offset_x += 5
    elif key == 'n':  # Next shirt image
        shirtImageNumber = (shirtImageNumber + 1) % len(listShirts)
    elif key == 'b':  # Previous shirt image
        shirtImageNumber = (shirtImageNumber - 1) % len(listShirts)
    elif key == 'm':  # Next pants image
        pantsImageNumber = (pantsImageNumber + 1) % len(listPants)
    elif key == 'v':  # Previous pants image
        pantsImageNumber = (pantsImageNumber - 1) % len(listPants)

    return '', 200

# Dynamic video feed route
@app.route('/video_feed/<category>')
def video_feed(category):
    return Response(generate_frames(category), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start voice command listening in a separate thread
    voice_thread = Thread(target=listen_for_commands, daemon=True)
    voice_thread.start()
    
    # Run Flask app
    app.run(debug=True)
