#!/usr/bin/env python3
"""
Standalone ONNX OBB visualizer

Purpose:
- Load an exported ONNX model and a test image
- Run inference with onnxruntime
- Visualize oriented bounding boxes (OBB) from the raw model outputs

Assumptions:
- Model input: [1, 3, 640, 640] in RGB, float32, 0..1
- Model output (Ultralytics OBB export): [1, 6, 8400]
  channels = [cx, cy, w, h, theta, score] (most exports use this order)

Two parse modes are provided:
- raw: treat channels directly as pixel units at 640 and theta in radians (if values > 2*pi, degrees are converted)
- decode: apply a YOLO-like decode heuristic (grids at 80/40/20 with strides 8/16/32 and sigmoid/tanh transforms)

NOTE: This script is a debugging tool to compare ONNX output parsing variants and
produce visual overlays. It does not change your model.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required. Activate venv and pip install onnxruntime") from exc

# Optional: PyTorch/Ultralytics for reference comparison
try:
    from ultralytics import YOLO as _Y
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@dataclass
class OBB:
    corners: np.ndarray  # shape (4,2) in pixel coords
    score: float


def letterbox_resize_bgr_to_rgb(image_bgr: np.ndarray, size: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    """Letterbox to square `size` keeping aspect ratio; return NCHW tensor and the letterboxed BGR image."""
    h, w = image_bgr.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None, ...]
    return chw, canvas


def run_onnx(model_path: Path, input_chw: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])  # CPU for reproducibility
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    out = sess.run(None, {inp_name: input_chw})
    out_arr = out[0]
    if out_arr.ndim != 3 or out_arr.shape[0] != 1:
        raise RuntimeError(f"Unexpected output shape: {out_arr.shape}")
    dims = list(out_arr.shape)
    return out_arr, dims  # [1, C, N]


def nms_aabb(boxes_xywh: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45, top_k: int = 300) -> List[int]:
    if len(scores) == 0:
        return []
    order = scores.argsort()[::-1]
    if top_k and len(order) > top_k:
        order = order[:top_k]
    keep: List[int] = []
    x = boxes_xywh[:, 0]
    y = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    x2 = x + w
    y2 = y + h
    areas = w * h
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def obb_from_raw_channels(ch: np.ndarray, conf_thr: float) -> List[OBB]:
    # ch: [6, 8400] expected as [cx,cy,w,h,theta,score] in pixel units
    if ch.shape[0] != 6:
        raise RuntimeError(f"Expected 6 channels, got {ch.shape[0]}")
    cx, cy, w, h, th, sc = ch
    # Heuristics for angle: if looks like degrees, convert
    th = np.where(np.abs(th) > 6.0, th * math.pi / 180.0, th)
    # Filter by score
    mask = sc >= conf_thr
    cx, cy, w, h, th, sc = cx[mask], cy[mask], w[mask], h[mask], th[mask], sc[mask]

    half_w = w / 2.0
    half_h = h / 2.0
    cos_t = np.cos(th)
    sin_t = np.sin(th)

    rel = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
    rel = rel[None, :, :] * np.stack([half_w, half_h], axis=1)[:, None, :]
    # Rotate and translate
    rot_x = rel[..., 0] * cos_t[:, None] - rel[..., 1] * sin_t[:, None]
    rot_y = rel[..., 0] * sin_t[:, None] + rel[..., 1] * cos_t[:, None]
    corners = np.stack([rot_x + cx[:, None], rot_y + cy[:, None]], axis=-1)  # [K,4,2]

    # AABB for NMS
    min_xy = corners.min(axis=1)
    max_xy = corners.max(axis=1)
    aabb_xywh = np.concatenate([min_xy, (max_xy - min_xy)], axis=1)
    keep = nms_aabb(aabb_xywh, sc, iou_thr=0.45, top_k=300)

    return [OBB(corners=corners[i].astype(np.float32), score=float(sc[i])) for i in keep]


def recover_anchor_mapping(ch: np.ndarray, input_size: int) -> np.ndarray:
    """Auto-recover ONNX anchor ordering by clustering cy/cx channels."""
    C, N = ch.shape
    if C != 6:
        raise RuntimeError(f"Expected 6 channels, got {C}")
    
    # Expected grid sizes for input_size
    strides = [8, 16, 32]
    sizes = [input_size // s for s in strides]
    expected_total = sum(n * n for n in sizes)
    
    if N != expected_total:
        raise RuntimeError(f"Anchor count mismatch: expected {expected_total}, got {N}")
    
    print(f"🔍 Recovering anchor mapping for {input_size}x{input_size} → grids {sizes}")
    
    # Extract raw cx, cy channels (before sigmoid)
    cx_raw, cy_raw = ch[0], ch[1]
    
    # Apply sigmoid to get normalized coordinates
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))
    cx_norm = sigmoid(cx_raw)  # [0,1] range
    cy_norm = sigmoid(cy_raw)  # [0,1] range
    
    # Convert to pixel space for clustering
    cx_pixel = cx_norm * input_size
    cy_pixel = cy_norm * input_size
    
    # Create mapping array: index -> (scale_idx, gy, gx)
    mapping = np.zeros((N, 3), dtype=np.int32)
    
    # Sort by cy first (row-major), then by cx (column-major within row)
    sort_indices = np.lexsort((cx_pixel, cy_pixel))
    
    offset = 0
    for scale_idx, size in enumerate(sizes):
        count = size * size
        scale_indices = sort_indices[offset:offset + count]
        
        # Within this scale, sort by grid position
        scale_cy = cy_pixel[scale_indices]
        scale_cx = cx_pixel[scale_indices]
        
        # Quantize to grid positions
        grid_y = np.round(scale_cy / input_size * size).astype(np.int32)
        grid_x = np.round(scale_cx / input_size * size).astype(np.int32)
        
        # Clip to valid range
        grid_y = np.clip(grid_y, 0, size - 1)
        grid_x = np.clip(grid_x, 0, size - 1)
        
        # Sort by grid position (row-major: gy * size + gx)
        grid_order = np.argsort(grid_y * size + grid_x)
        ordered_indices = scale_indices[grid_order]
        
        # Fill mapping
        for i, orig_idx in enumerate(ordered_indices):
            gy = i // size
            gx = i % size
            mapping[orig_idx] = [scale_idx, gy, gx]
        
        offset += count
        print(f"  Scale {scale_idx}: {size}x{size} grid, stride {strides[scale_idx]}")
    
    return mapping


def obb_from_yolo_decode(ch: np.ndarray, conf_thr: float, input_size: int, mapping: np.ndarray = None) -> List[OBB]:
    # Apply YOLO-like decode heuristic to 3 scales based on input size
    if ch.shape[0] != 6:
        raise RuntimeError(f"Expected 6 channels, got {ch.shape[0]}")
    C, N = ch.shape
    
    # Use recovered mapping if provided, otherwise use standard ordering
    if mapping is not None:
        print("🔧 Using recovered anchor mapping")
    else:
        # build grids from the actual model input size (e.g., 1088 -> 136,68,34)
        strides = [8, 16, 32]
        sizes = [input_size // s for s in strides]
        if sum(n * n for n in sizes) != N:
            raise RuntimeError(f"Anchor count mismatch: expected {sum(n*n for n in sizes)}, got {N}")
    
    cx_raw, cy_raw, w_raw, h_raw, th_raw, sc_raw = ch

    # sigmoid/tanh helpers
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))
    tanh = np.tanh

    out: List[OBB] = []
    
    if mapping is not None:
        # Use recovered mapping
        strides = [8, 16, 32]
        sizes = [input_size // s for s in strides]
        
        for scale_idx, size in enumerate(sizes):
            stride = strides[scale_idx]
            
            # Find all anchors for this scale
            scale_mask = mapping[:, 0] == scale_idx
            scale_indices = np.where(scale_mask)[0]
            
            if len(scale_indices) == 0:
                continue
                
            # Get grid positions from mapping
            gx = mapping[scale_indices, 2].astype(np.float32)
            gy = mapping[scale_indices, 1].astype(np.float32)
            
            # Apply transforms
            cx = (sigmoid(cx_raw[scale_indices]) * 2.0 - 0.5 + gx) * stride
            cy = (sigmoid(cy_raw[scale_indices]) * 2.0 - 0.5 + gy) * stride
            w  = (sigmoid(w_raw[scale_indices]) * 2.0) ** 2 * stride
            h  = (sigmoid(h_raw[scale_indices]) * 2.0) ** 2 * stride
            th = tanh(th_raw[scale_indices]) * math.pi
            sc = sigmoid(sc_raw[scale_indices])

            mask = sc >= conf_thr
            if not np.any(mask):
                continue
            cx, cy, w, h, th, sc = cx[mask], cy[mask], w[mask], h[mask], th[mask], sc[mask]

            half_w = w / 2.0
            half_h = h / 2.0
            cos_t = np.cos(th)
            sin_t = np.sin(th)
            rel = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
            rel = rel[None, :, :] * np.stack([half_w, half_h], axis=1)[:, None, :]
            rot_x = rel[..., 0] * cos_t[:, None] - rel[..., 1] * sin_t[:, None]
            rot_y = rel[..., 0] * sin_t[:, None] + rel[..., 1] * cos_t[:, None]
            corners = np.stack([rot_x + cx[:, None], rot_y + cy[:, None]], axis=-1)  # [K,4,2]

            min_xy = corners.min(axis=1)
            max_xy = corners.max(axis=1)
            aabb_xywh = np.concatenate([min_xy, (max_xy - min_xy)], axis=1)
            keep = nms_aabb(aabb_xywh, sc, iou_thr=0.45, top_k=300)
            out.extend([OBB(corners=corners[i].astype(np.float32), score=float(sc[i])) for i in keep])
    else:
        # Standard ordering
        strides = [8, 16, 32]
        sizes = [input_size // s for s in strides]
        offset = 0
        for size, stride in zip(sizes, strides):
            count = size * size
            sl = slice(offset, offset + count)
            offset += count
            gx, gy = np.meshgrid(np.arange(size), np.arange(size))  # [H,W]
            gx = gx.reshape(-1)
            gy = gy.reshape(-1)

            cx = (sigmoid(cx_raw[sl]) * 2.0 - 0.5 + gx) * stride
            cy = (sigmoid(cy_raw[sl]) * 2.0 - 0.5 + gy) * stride
            w  = (sigmoid(w_raw[sl]) * 2.0) ** 2 * stride
            h  = (sigmoid(h_raw[sl]) * 2.0) ** 2 * stride
            th = tanh(th_raw[sl]) * math.pi
            sc = sigmoid(sc_raw[sl])

            mask = sc >= conf_thr
            if not np.any(mask):
                continue
            cx, cy, w, h, th, sc = cx[mask], cy[mask], w[mask], h[mask], th[mask], sc[mask]

            half_w = w / 2.0
            half_h = h / 2.0
            cos_t = np.cos(th)
            sin_t = np.sin(th)
            rel = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
            rel = rel[None, :, :] * np.stack([half_w, half_h], axis=1)[:, None, :]
            rot_x = rel[..., 0] * cos_t[:, None] - rel[..., 1] * sin_t[:, None]
            rot_y = rel[..., 0] * sin_t[:, None] + rel[..., 1] * cos_t[:, None]
            corners = np.stack([rot_x + cx[:, None], rot_y + cy[:, None]], axis=-1)  # [K,4,2]

            min_xy = corners.min(axis=1)
            max_xy = corners.max(axis=1)
            aabb_xywh = np.concatenate([min_xy, (max_xy - min_xy)], axis=1)
            keep = nms_aabb(aabb_xywh, sc, iou_thr=0.45, top_k=300)
            out.extend([OBB(corners=corners[i].astype(np.float32), score=float(sc[i])) for i in keep])

    return out


def draw_obb(image_bgr: np.ndarray, obbs: List[OBB]) -> np.ndarray:
    out = image_bgr.copy()
    for obb in obbs:
        pts = obb.corners.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        # draw score near first corner
        p = pts[0]
        cv2.putText(out, f"{obb.score:.2f}", (int(p[0]), int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out


def load_obb_labels(label_path: Path, img_w: int, img_h: int) -> List[np.ndarray]:
    boxes: List[np.ndarray] = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            coords = list(map(float, parts[1:]))
            pts = []
            for i in range(0, 8, 2):
                x = coords[i] * img_w
                y = coords[i+1] * img_h
                pts.append([x, y])
            boxes.append(np.array(pts, dtype=np.float32))
    return boxes


def letterbox_points(pts: np.ndarray, orig_w: int, orig_h: int, size: int) -> np.ndarray:
    r = min(size / orig_w, size / orig_h)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    out = pts.copy()
    out[:, 0] = pts[:, 0] * r + pad_x
    out[:, 1] = pts[:, 1] * r + pad_y
    return out


def run_torch_and_visualize(weights: Path, image_path: Path, conf: float, out_path: Path) -> Tuple[int, float]:
    if not _HAS_TORCH:
        raise SystemExit("Ultralytics not available in this environment; install to use --torch")
    m = _Y(str(weights))
    res = m(str(image_path), conf=conf, iou=0.7, imgsz=640, verbose=False)[0]
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    if res.obb is None or len(res.obb.xyxyxyxy) == 0:
        cv2.imwrite(str(out_path), img)
        return 0, 0.0
    boxes = res.obb.xyxyxyxy.cpu().numpy()  # [K,4,2]
    scores = res.obb.conf.cpu().numpy()
    obbs = [OBB(corners=boxes[i].astype(np.float32), score=float(scores[i])) for i in range(boxes.shape[0])]
    vis = draw_obb(img, obbs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return len(obbs), float(scores.mean())


def main():
    ap = argparse.ArgumentParser(description="ONNX OBB output visualizer")
    ap.add_argument("--model", required=True, help="Path to ONNX model")
    ap.add_argument("--image", required=True, help="Path to test image")
    ap.add_argument("--mode", choices=["raw", "decode"], default="raw", help="Parse mode")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--out", type=str, default="outputs/onnx_debug/result.jpg", help="Output image path")
    ap.add_argument("--labels", type=str, default="", help="Optional YOLO OBB label file to overlay/compare")
    ap.add_argument("--torch", type=str, default="", help="Optional PyTorch .pt weights to compare")
    ap.add_argument("--tune", action="store_true", help="Grid-search decode constants to match PyTorch")
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    # Match model input size dynamically
    sess = ort.InferenceSession(str(Path(args.model)), providers=["CPUExecutionProvider"]) 
    ishp = sess.get_inputs()[0].shape  # [1,3,H,W]
    size = int(ishp[2]) if isinstance(ishp[2], (int,)) else 1088
    chw, vis_base = letterbox_resize_bgr_to_rgb(img_bgr, size=size)
    out_arr, dims = run_onnx(Path(args.model), chw)
    print("Output shape:", dims)

    # Convert to [C,N]
    ch = out_arr[0]
    C, N = ch.shape
    print(f"Channels: {C}, Anchors: {N}")

    if args.mode == "raw":
        obbs = obb_from_raw_channels(ch, conf_thr=args.conf)
    else:
        obbs = obb_from_yolo_decode(ch, conf_thr=args.conf, input_size=size)

    print(f"Detections kept (ONNX {args.mode}): {len(obbs)}")
    out_img = draw_obb(vis_base, obbs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img)
    print(f"Saved visualization to: {out_path}")

    # Optional label overlay and simple IoU stats (AABB approximation)
    if args.labels:
        h0, w0 = img_bgr.shape[:2]
        lbls = load_obb_labels(Path(args.labels), w0, h0)
        if lbls:
            # letterbox labels to the same canvas size
            lbls_lb = [letterbox_points(b, w0, h0, size) for b in lbls]
            canvas = vis_base.copy()
            for b in lbls_lb:
                b_int = b.astype(np.int32)
                cv2.polylines(canvas, [b_int], True, (0,0,255), 2)
            cmp_path = out_path.with_name(out_path.stem + "_labels.jpg")
            cv2.imwrite(str(cmp_path), canvas)
            print(f"Saved labels overlay to: {cmp_path}")

    # Optional torch comparison
    if args.torch:
        t_out = out_path.with_name(out_path.stem + "_torch.jpg")
        dets, avg_conf = run_torch_and_visualize(Path(args.torch), Path(args.image), args.conf, t_out)
        print(f"PyTorch reference: {dets} detections, avg conf {avg_conf:.3f}. Saved: {t_out}")

    # Optional tuning: brute-force constants to reduce FP by matching torch
    if args.torch and args.tune and args.mode == "decode":
        print("\n🔧 Tuning decode constants against PyTorch...")
        # load torch OBBs as reference
        m = _Y(args.torch)
        ref = m(args.image, conf=args.conf, iou=0.7, imgsz=640, verbose=False)[0]
        if ref.obb is None or len(ref.obb) == 0:
            print("No torch detections; skip tuning")
            return
        ref_boxes = ref.obb.xyxyxyxy.cpu().numpy()
        # simple score: sum of max IoU per ref box against candidate boxes
        def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
            # approximate via AABB IoU to keep deps light
            ax1, ay1 = a[:,0].min(), a[:,1].min(); ax2, ay2 = a[:,0].max(), a[:,1].max()
            bx1, by1 = b[:,0].min(), b[:,1].min(); bx2, by2 = b[:,0].max(), b[:,1].max()
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = inter_w * inter_h
            area_a = (ax2-ax1)*(ay2-ay1)
            area_b = (bx2-bx1)*(by2-by1)
            if area_a <= 0 or area_b <= 0: return 0.0
            return inter / (area_a + area_b - inter + 1e-9)
        best = (0.0, None)
        # candidate ranges
        ac_list = [1.5, 2.0, 2.5]
        bc_list = [0.0, 0.25, 0.5, 0.75]
        aw_list = [1.5, 2.0, 2.5]
        th_scale = [math.pi/2, math.pi, math.pi*1.5]
        for ac in ac_list:
            for bc in bc_list:
                for aw in aw_list:
                    for ts in th_scale:
                        # decode with candidates
                        C, N = ch.shape
                        strides = [8,16,32]; sizes=[80,40,20]
                        cx_raw, cy_raw, w_raw, h_raw, th_raw, sc_raw = ch
                        sigmoid = lambda v: 1.0/(1.0+np.exp(-v))
                        out_boxes: List[np.ndarray] = []
                        offset=0
                        for size,stride in zip(sizes,strides):
                            count=size*size
                            gx, gy = np.meshgrid(np.arange(size), np.arange(size))
                            gx=gx.reshape(-1); gy=gy.reshape(-1)
                            sl = slice(offset, offset+count)
                            cx = (sigmoid(cx_raw[sl])*ac - bc + gx)*stride
                            cy = (sigmoid(cy_raw[sl])*ac - bc + gy)*stride
                            w  = (sigmoid(w_raw[sl])*aw)**2*stride
                            h  = (sigmoid(h_raw[sl])*aw)**2*stride
                            th = np.tanh(th_raw[sl])*ts
                            sc = sigmoid(sc_raw[sl])
                            mask = sc >= args.conf
                            cx,cy,w,h,th = cx[mask],cy[mask],w[mask],h[mask],th[mask]
                            half_w=w/2; half_h=h/2
                            cos=np.cos(th); sin=np.sin(th)
                            rel=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],dtype=np.float32)
                            rel=rel[None,:,:]*np.stack([half_w,half_h],axis=1)[:,None,:]
                            rx=rel[...,0]*cos[:,None]-rel[...,1]*sin[:,None]
                            ry=rel[...,0]*sin[:,None]+rel[...,1]*cos[:,None]
                            corners=np.stack([rx+cx[:,None], ry+cy[:,None]],axis=-1)
                            out_boxes.extend([corners[i] for i in range(corners.shape[0])])
                            offset+=count
                        # score IoU sum
                        total=0.0
                        for rb in ref_boxes:
                            best_iou=0.0
                            for cb in out_boxes[:500]:
                                best_iou=max(best_iou, poly_iou(rb, cb))
                            total += best_iou
                        if total>best[0]:
                            best=(total,(ac,bc,aw,ts))
        if best[1]:
            ac,bc,aw,ts=best[1]
            print(f"Best decode constants: ac={ac} bc={bc} aw={aw} theta_scale={ts:.3f}")


if __name__ == "__main__":
    main()


