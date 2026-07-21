import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Simulate an aerial target (e.g., a runway intersection viewed from two different angles)
# View A: Drone approaching straight on
view_A = np.zeros((400, 400), dtype=np.uint8)
cv2.line(view_A, (100, 200), (300, 200), 255, 10) # Runway 1
cv2.line(view_A, (200, 100), (200, 300), 255, 10) # Intersecting Taxiway

# View B: Drone rotated 45 degrees and closer (Simulating flight trajectory shift)
view_B = np.zeros((400, 400), dtype=np.uint8)
M = cv2.getRotationMatrix2D((200, 200), 45, 1.2)
view_B = cv2.warpAffine(view_A, M, (400, 400))

# 2. Initialize the Feature Extractor (ORB acts as our localized frontend descriptor)
orb = cv2.ORB_create(nfeatures=500)

# Extract keypoints (locations) and descriptors (mathematical vectors describing the texture)
kpA, desA = orb.detectAndCompute(view_A, None)
kpB, desB = orb.detectAndCompute(view_B, None)

# 3. Build a Brute-Force Matcher with Hamming Distance to associate data across frames
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desA, desB)

# Sort them based on distance (confidence scores)
matches = sorted(matches, key=lambda x: x.distance)

# 4. Render the Data Association Link Matrix
matched_visual = cv2.drawMatches(view_A, kpA, view_B, kpB, matches[:15], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("==================================================")
print("          SLAM DATA ASSOCIATION ENGAGED           ")
print("==================================================")
print(f"Features Extracted in View A: {len(kpA)}")
print(f"Features Extracted in View B: {len(kpB)}")
print(f"Valid Spatial Correspondences Established: {len(matches)}")
print("==================================================")

plt.figure(figsize=(12, 6))
plt.title("Deep SLAM Frontend: Verifying Feature Correspondences Across Disparate Sensor Views")
plt.imshow(matched_visual)
plt.axis('off')
plt.show()