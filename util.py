import numpy as np
import cv2 as cv
import supervision as sv

def bbox_checking(region, bbox):
    """Return True if bbox center is inside region."""

    def _as_np(a, dtype=np.float32):
        return np.asarray(a, dtype=dtype)

    def _rect_xyxy_to_center(xyxy):
        x1, y1, x2, y2 = map(float, xyxy)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    def _obb_params_to_center(p):
        cx, cy, _, _, _ = map(float, p)
        return cx, cy

    def _polygon_to_center(poly):
        poly = _as_np(poly)
        cx = np.mean(poly[:, 0])
        cy = np.mean(poly[:, 1])
        return cx, cy

    def _bbox_to_center(b):
        b = _as_np(b)
        if b.shape == (4,):   # xyxy
            return _rect_xyxy_to_center(b)
        if b.shape == (5,):   # obb params
            return _obb_params_to_center(b)
        if b.ndim == 2 and b.shape[1] == 2:  # polygon
            return _polygon_to_center(b)
        raise ValueError("Invalid bbox format")

    def _region_to_poly(r):
        r = _as_np(r)
        if r.shape == (4,):
            x1, y1, x2, y2 = r
            return np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
        if r.ndim == 2 and r.shape[1] == 2:
            return r.astype(np.float32)
        raise ValueError("Invalid region format")

    # Convert
    region_poly = _region_to_poly(region)
    cx, cy = _bbox_to_center(bbox)

    region_cnt = region_poly.reshape((-1, 1, 2)).astype(np.float32)

    # Check center inside region
    return cv.pointPolygonTest(region_cnt, (float(cx), float(cy)), False) >= 0

def draw_bbox(
    frame,
    box,
    label,
    text_color=sv.Color.BLACK,
    text_bg_color=sv.Color.WHITE,
    bbox_color=sv.Color.BLUE,
    thickness=1,
):
    # ========================
    # 🔷 1. Xử lý bbox / OBB
    # ========================
    box = np.asarray(box)

    if box.shape == (4,):  # xyxy
        x1, y1, x2, y2 = map(int, box)
        pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)

    elif box.shape == (4, 2):  # OBB polygon
        pts = box.astype(np.int32)

    else:
        raise ValueError("bbox phải là (4,) hoặc (4,2)")

    pts = pts.astype(np.int32)

    # 1. Vẽ bbox / polygon
    cv.polylines(
        frame,
        [pts],
        isClosed=True,
        color=bbox_color.as_bgr(),
        thickness=thickness
    )

    # 2. Vẽ label nền trắng
    x_text, y_text = pts[0]

    (w, h), _ = cv.getTextSize(
        label,
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )

    # tránh vẽ ra ngoài frame
    y_text = max(y_text, h + 4)

    # nền trắng
    cv.rectangle(
        frame,
        (x_text, y_text - h - 4),
        (x_text + w, y_text),
        text_bg_color.as_bgr(),
        -1
    )

    # chữ
    cv.putText(
        frame,
        label,
        (x_text, y_text - 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color.as_bgr(),
        1,
        cv.LINE_AA
    )


def display(
    frame,
    bboxes,
    labels,
    text_color=sv.Color.BLACK,
    text_bg_color=sv.Color.WHITE,
    bbox_color=sv.Color.BLUE,
    thickness=1,
):
    # annotated_frame = frame.copy()
    bboxes = np.asarray(bboxes, dtype=object)  # giữ được (4,) và (4,2)
    # print("bboxes: ", bboxes.shape)
    # 🔷 chuẩn hóa màu
    if not isinstance(bbox_color, list):
        bbox_color = [bbox_color] * len(bboxes)

    for i, (box, label) in enumerate(zip(bboxes, labels)):
        color = bbox_color[i % len(bbox_color)]
        # Draw
        draw_bbox(
            frame,
            box,
            label,
            text_color=text_color,
            text_bg_color=text_bg_color,
            bbox_color=color,
            thickness=thickness
        )

    # return annotated_frame

def get_colors(bboxes, region, colors=None):
    COLORS = {"in": sv.Color.GREEN, "out": sv.Color.RED} if colors is None else colors
    colors = []
    for obj in bboxes:
        if bbox_checking(region, obj):
            colors.append(COLORS["in"])
        else:
            colors.append(COLORS["out"])
    return colors