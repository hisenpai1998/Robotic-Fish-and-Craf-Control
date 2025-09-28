import cv2
import numpy as np
import math

# ======= SETTINGS =======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
USE_HSV = True          # True = color segmentation     | False = adaptive grayscale
ROI_FRACTION = 1.0      # e.g. 0.7 to crop center (reduce background effect)
SHOW_DEBUG = True

# If using color: set HSV range for your spiral color
# Example for BLUE; adjust with live histogram if needed
HSV_LO = (100, 80, 60)
HSV_HI = (135, 255, 255)

# ======= HELPERS =======
def apply_gamma(img, gamma=1.2):
    inv = 1.0 / max(gamma, 1e-6)
    tbl = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, tbl)

def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def get_roi(frame, frac=1.0):
    if frac >= 0.999: return frame, (0,0,1.0,1.0)
    H, W = frame.shape[:2]
    w = int(W*frac); h = int(H*frac)
    x0 = (W - w)//2; y0 = (H - h)//2
    return frame[y0:y0+h, x0:x0+w], (x0, y0, w, h)

def largest_good_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 400:  # ignore tiny blobs
            continue
        p = cv2.arcLength(c, True) + 1e-6
        circularity = 4*math.pi*area/(p*p)  # ~1 for circle
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area/hull_area          # ~1 for filled shapes
        # Score: prefer compact, not-too-spiky blobs
        score = 0.6*circularity + 0.4*solidity
        if score > best_score:
            best_score = score; best = c
    return best

def centroid_of(c):
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

# ======= CAPTURE =======
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
# Try to reduce auto flicker
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.75 auto (varies by driver), 0.25 manual-ish
# cap.set(cv2.CAP_PROP_EXPOSURE, -5)       # optional: tune manually if your camera supports

while True:
    ok, frame = cap.read()
    if not ok: break

    # Normalize brightness/contrast a bit
    frame = apply_gamma(frame, 1.2)

    # Optional ROI to reduce background influence
    roi, (x0,y0,w,h) = get_roi(frame, ROI_FRACTION)

    if USE_HSV:
        # ---------- COLOR MODE ----------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)
    else:
        # ---------- ADAPTIVE GRAYSCALE MODE ----------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = clahe_gray(gray)                       # stabilize illumination
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)

    # Find the best contour (likely the spiral blob)
    c = largest_good_contour(mask)
    if c is not None:
        # Draw contour offset back to full frame coords
        c_up = c + np.array([[[x0,y0]]], dtype=np.int32)
        cv2.drawContours(frame, [c_up], -1, (0,200,50), 2)
        ctr = centroid_of(c)
        if ctr:
            cx, cy = ctr[0]+x0, ctr[1]+y0
            cv2.circle(frame, (cx,cy), 6, (0,0,255), -1)
            if SHOW_DEBUG:
                cv2.putText(frame, f"Centroid: {cx},{cy}", (cx+8, cy-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,220,255), 1, cv2.LINE_AA)

    # Show
    cv2.imshow("Spiral robust detection", frame)
    if SHOW_DEBUG:
        cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
