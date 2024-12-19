from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 자세 판별 함수
def check_posture(landmarks, image_height, image_width):
    """
    상반신 랜드마크를 기반으로 거북목과 상체 구부림(허리디스크 위험)을 판별.
    """
    # 랜드마크 추출
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # 1. 거북목 분석 (Neck Tilt)
    neck_tilt = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2) * image_width

    # 2. 상체 구부림 분석 (Forward Lean)
    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    forward_lean = abs(nose.y - shoulder_avg_y) * image_height

    return neck_tilt, forward_lean

def check_posture(landmarks, image_height, image_width):
    """
    상반신 랜드마크를 기반으로 거북목과 상체 구부림(허리디스크 위험)을 판별.
    """
    # 랜드마크 추출
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    neck_tilt = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2) * image_width
    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    forward_lean = abs(nose.y - shoulder_avg_y) * image_height

    return neck_tilt, forward_lean

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe 처리
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        image_height, image_width, _ = frame.shape

        if results.pose_landmarks:
            # 자세 분석
            neck_tilt, forward_lean = check_posture(results.pose_landmarks.landmark, image_height, image_width)

            # 거북목 경고 기준 수정
            if neck_tilt > 130:
                cv2.putText(
                    frame, f"Forward Head! Tilt: {int(neck_tilt)}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
            else:
                cv2.putText(
                    frame, f"Good Head Posture! Tilt: {int(neck_tilt)}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

            # 상체 구부림 경고 기준 수정
            if forward_lean < 170:
                cv2.putText(
                    frame, f"Forward Lean! Lean: {int(forward_lean)}", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
            else:
                cv2.putText(
                    frame, f"Good Upper Body Posture! Lean: {int(forward_lean)}", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

            # 랜드마크 및 연결선 시각화
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # 프레임 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
