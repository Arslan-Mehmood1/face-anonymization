
import numpy as np
import cv2
import skimage

def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)

def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = False,
        ovcolor: tuple = (0, 0, 0),
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox

def anonymize_face(
        face_detections, frame, mask_scale,model_name,
        replacewith='blur', ellipse='True'
):
    for i, face in enumerate(face_detections):
        if model_name=='centerface':
            boxes, score = face[:4], face[4]
            x1, y1, x2, y2 = boxes.astype(int)
            x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        else:
            # print("face : ",face)
            x, y, w, h = face
            x1, y1, x2, y2 = scale_bb(x, y, x + w, y + h, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
        draw_det(
            frame, 0.0, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse
        )
