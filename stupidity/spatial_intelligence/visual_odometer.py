import numpy as np
import cv2

# 1. SIMULATE FRAME 1 (Aircraft looks down at a target zone)
# We create a black frame and draw some synthetic ground features (mock buildings/rocks)
frame_1 = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.rectangle(frame_1, (300, 200), (350, 250), (255, 255, 255), -1) # Feature A
cv2.circle(frame_1, (800, 500), 20, (255, 255, 255), -1)          # Feature B
cv2.polylines(frame_1, [np.array([[900,100], [950,150], [850,200]])], True, (255,255,255), 2) # Feature C

# 2. SIMULATE FRAME 2 (Aircraft has flown forward and slightly right)
# All features have shifted down and left on the camera sensor screen
frame_2 = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.rectangle(frame_2, (280, 240), (330, 290), (255, 255, 255), -1) # Feature A moved
cv2.circle(frame_2, (780, 540), 20, (255, 255, 255), -1)          # Feature B moved
cv2.polylines(frame_2, [np.array([[880,140], [930,190], [830,240]])], True, (255,255,255), 2) # Feature C moved

# Convert frames to grayscale for processing
gray1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

# 3. DETECT KEYPOINTS IN FRAME 1 USING ORB
# GoodFeaturesToTrack finds sharp corners mathematically optimized for motion tracking
points_frame1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# 4. TRACK THE POINTS INTO FRAME 2 USING OPTICAL FLOW (Lucas-Kanade)
points_frame2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, points_frame1, None)

# Filter out only the successfully tracked points
valid_old = points_frame1[status == 1]
valid_new = points_frame2[status == 1]

# 5. CALCULATE THE AVERAGE VELOCITY VECTOR (Pixel Shift)
pixel_shifts = valid_new - valid_old
avg_dx = np.mean(pixel_shifts[:, 0])
avg_dy = np.mean(pixel_shifts[:, 1])

print("==================================================")
print("        VISUAL ODOMETRY MOTION DETECTION          ")
print("==================================================")
print(f"Successfully tracking {len(valid_new)} environmental ground assets.")
print(f"Average Pixel Shift Delta X (Horizontal): {avg_dx:.2f} pixels")
print(f"Average Pixel Shift Delta Y (Vertical):   {avg_dy:.2f} pixels")
print("--------------------------------------------------")

# Interpret direction
if avg_dy > 0:
    print("AI Telemetry Inference: Aircraft is moving FORWARD/DOWN relative to terrain.")
if avg_dx < 0:
    print("AI Telemetry Inference: Aircraft is drifting RIGHT (causing ground to move left).")
print("==================================================")