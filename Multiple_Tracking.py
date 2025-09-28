import cv2
import numpy as np
import math, time, csv
from collections import deque

# ================== USER SETTINGS ==================
STREAM = 0                       # CAM index or IP URL (e.g., "rtsp://...")
FRAME_W, FRAME_H = 640, 480

# Balloon real diameter (meters) and calibrated focal length in pixels
D_REAL_M  = 0.30                 # e.g., 30 cm
FOCAL_PIX = 950.0                # calibrate once (see notes)

# HSV range (example: BLUE). For RED, use two ranges and OR them.
HSV_LO = (100, 120, 80)
HSV_HI = (135, 255, 255)

MIN_AREA     = 250               # ignore tiny blobs
ARROW_PIX    = 80                # direction arrow length
TEXT_COLOR   = (0, 0, 255)       # BGR
DRAW_COLOR   = (255, 0, 0)       # contour color
CENTER_COLOR = (0, 255, 0)

# Multi-object tracking
MAX_TRACKS         = 50
ASSOC_DIST_THRESH  = 60.0         # pixels: max distance to re-associate
HIST_LEN           = 20           # history buffer per track
FORGET_SECONDS     = 1.0          # drop track if unseen this long

# Logging (set None to disable)
CSV_PATH = "C:\\Users\\Mezin\\OneDrive - Högskolan i Halmstad\\Skrivbordet\\Pose_log_MT.csv"   # e.g., "multi_balloon_log.csv"

# ================== HELPERS ==================
def find_blobs(mask, min_area=MIN_AREA):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: 
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(x)
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(y)
        d_pix = int(2*r)
        blobs.append({"contour": c, "cx": cx, "cy": cy, "d_pix": d_pix})
    return blobs

def z_from_dpix(d_pix, D_real, f_pix):
    if d_pix <= 1: return None
    return (f_pix * D_real) / float(d_pix)

def unit(vx, vy):
    n = math.hypot(vx, vy)
    return (0.0, 0.0) if n < 1e-6 else (vx/n, vy/n)

# Simple nearest-neighbor data association
class Track:
    def __init__(self, tid, cx, cy, d_pix, z_m, t, hist_len):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.d_pix = d_pix
        self.z_m = z_m
        self.last_t = t
        self.history = deque(maxlen=hist_len)  # (t, x, y, z, d)

    def update(self, cx, cy, d_pix, z_m, t):
        self.cx, self.cy = cx, cy
        self.d_pix = d_pix
        self.z_m = z_m
        self.last_t = t
        self.history.append((t, cx, cy, z_m, d_pix))

def associate_tracks(tracks, blobs, t, dist_thresh):
    """
    Greedy nearest neighbor: match each blob to closest track under threshold.
    Unmatched blobs -> new tracks. Unseen tracks are kept; caller can prune by time.
    """
    used_tracks = set()
    used_blobs = set()
    # Precompute distances
    dists = []
    for bi, b in enumerate(blobs):
        for ti, tr in enumerate(tracks):
            d = math.hypot(b["cx"] - tr.cx, b["cy"] - tr.cy)
            dists.append((d, ti, bi))
    dists.sort(key=lambda x: x[0])

    # Assign greedily
    for d, ti, bi in dists:
        if d > dist_thresh: 
            continue
        if ti in used_tracks or bi in used_blobs:
            continue
        tr = tracks[ti]
        b  = blobs[bi]
        z_m = z_from_dpix(b["d_pix"], D_REAL_M, FOCAL_PIX)
        tr.update(b["cx"], b["cy"], b["d_pix"], z_m, t)
        used_tracks.add(ti)
        used_blobs.add(bi)

    # Create new tracks for unmatched blobs
    next_id = (max([tr.id for tr in tracks], default=0) + 1) if tracks else 1
    for bi, b in enumerate(blobs):
        if bi in used_blobs:
            continue
        if len(tracks) >= MAX_TRACKS:
            break
        z_m = z_from_dpix(b["d_pix"], D_REAL_M, FOCAL_PIX)
        tr = Track(next_id, b["cx"], b["cy"], b["d_pix"], z_m, t, HIST_LEN)
        tr.history.append((t, b["cx"], b["cy"], z_m, b["d_pix"]))
        tracks.append(tr)
        next_id += 1

# ================== MAIN ==================
cap = cv2.VideoCapture(STREAM, cv2.CAP_DSHOW if isinstance(STREAM, int) else 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

tracks = []  # list[Track]

writer = None
csv_file = None
if CSV_PATH:
    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["t", "track_id", "x_pix", "y_pix", "z_m", "d_pix"])

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    now = time.time()

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # If RED balloon, combine two ranges (uncomment + set ranges):
    # mask1 = cv2.inRange(hsv, (0,120,80), (10,255,255))
    # mask2 = cv2.inRange(hsv, (170,120,80), (180,255,255))
    # mask = cv2.bitwise_or(mask1, mask2)

    # Single-range example (BLUE):
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)

    blobs = find_blobs(mask, MIN_AREA)

    # Associate/update tracks
    associate_tracks(tracks, blobs, now, ASSOC_DIST_THRESH)

    # Prune stale tracks
    tracks = [tr for tr in tracks if (now - tr.last_t) <= FORGET_SECONDS]

    # Draw and log
    for tr in tracks:
        # Direction arrow from motion over a short baseline
        if len(tr.history) >= 5:
            _, x0, y0, _, _ = tr.history[-5]
            vx, vy = tr.cx - x0, tr.cy - y0
            ux, uy = unit(vx, vy)
            end = (int(tr.cx + ARROW_PIX*ux), int(tr.cy + ARROW_PIX*uy))
            cv2.arrowedLine(frame, (tr.cx, tr.cy), end, (0, 255, 255), 3, tipLength=0.25)

        # Contour: draw the closest blob (approximate). For multi-blob we already used nearest neighbor,
        # so just draw a small circle + text
        cv2.circle(frame, (tr.cx, tr.cy), 6, CENTER_COLOR, -1)
        cv2.putText(frame, f"ID {tr.id}  Z={tr.z_m:.2f}m" if tr.z_m else f"ID {tr.id}",
                    (tr.cx+8, tr.cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

        if writer:
            writer.writerow([tr.last_t, tr.id, tr.cx, tr.cy, f"{tr.z_m:.3f}" if tr.z_m else "", tr.d_pix])

    cv2.imshow("Multi-object color tracking (pos, scale->Z, direction)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if writer:
    csv_file.close()
cap.release()
cv2.destroyAllWindows()
