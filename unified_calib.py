#!/usr/bin/env python3
"""
TRUE ONLINE Camera-LiDAR Extrinsic Calibration
Edge-based alignment - FIXED OPTIMIZER
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.optimize import minimize, differential_evolution
from collections import deque
import time, json, struct
from numba import njit, prange

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

@dataclass
class CameraConfig:
    fx: float = 1501.9374712879626; fy: float = 1498.8879775647906
    cx: float = 566.5690420612353; cy: float = 537.1294320963829
    width: int = 1224; height: int = 1024
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.array([-0.2306, 0.207, 0.0005, -0.002]))
    
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

@dataclass
class OnlineCalibConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    initial_T: np.ndarray = field(default_factory=lambda: np.array([
        [0.99919851,  0.04002921,  0.00000000,   0.15],
        [0.00000000,  0.00000000, -1.00000000,  -0.2815789473684212],
        [-0.04002921, 0.99919851,  0.00000000,  -0.13157894736842124],
        [0.0,         0.0,         0.0,          1.0]
    ], dtype=float))
    update_alpha: float = 0.5
    max_rotation_update: float = 0.02
    max_translation_update: float = 0.01
    # Offset injection: [x, y, z] in meters for translation, degrees for rotation
    trans_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rot_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

@njit(cache=True)
def _project_points_fast(xyz, R, t, fx, fy, cx, cy):
    n = len(xyz)
    uv = np.empty((n, 2), dtype=np.float32)
    depths = np.empty(n, dtype=np.float32)
    for i in range(n):
        px = R[0,0]*xyz[i,0] + R[0,1]*xyz[i,1] + R[0,2]*xyz[i,2] + t[0]
        py = R[1,0]*xyz[i,0] + R[1,1]*xyz[i,1] + R[1,2]*xyz[i,2] + t[1]
        pz = R[2,0]*xyz[i,0] + R[2,1]*xyz[i,1] + R[2,2]*xyz[i,2] + t[2]
        depths[i] = pz
        if pz > 0.3:
            uv[i, 0] = fx * px / pz + cx
            uv[i, 1] = fy * py / pz + cy
        else:
            uv[i, 0] = -1
            uv[i, 1] = -1
    return uv, depths

@njit(parallel=True, cache=True)
def _render_lidar_image(uv, depths, intensity, h, w, radius=2):
    img = np.zeros((h, w), dtype=np.float32)
    depth_buf = np.full((h, w), 1e10, dtype=np.float32)
    for i in prange(len(uv)):
        u, v, d = int(uv[i, 0]), int(uv[i, 1]), depths[i]
        if d <= 0.3 or u < 0 or v < 0 or u >= w or v >= h:
            continue
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                nu, nv = u + du, v + dv
                if 0 <= nu < w and 0 <= nv < h and d < depth_buf[nv, nu]:
                    depth_buf[nv, nu] = d
                    img[nv, nu] = intensity[i]
    return img

class OnlineCalibrator:
    def __init__(self, config):
        self.config = config
        self.T = config.initial_T.copy()
        self.T_initial = config.initial_T.copy()
        self.K = config.camera.K
        self.h, self.w = config.camera.height, config.camera.width
        
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, config.camera.dist_coeffs, None, self.K,
            (self.w, self.h), cv2.CV_32FC1)
        
        self.frame_count = 0
        self.update_count = 0
        self.edge_costs = deque(maxlen=100)
        
        self.offset_injected = False
        self.offset_injection_frame = -1
        self.T_at_injection = None
        
        self.last_dT = np.zeros(3)
        self.last_dR = np.zeros(3)
        self.last_opt_result = None
        
        # Warmup numba
        dummy = np.zeros((10, 3), dtype=np.float32)
        _project_points_fast(dummy, np.eye(3), np.zeros(3), 1000, 1000, 500, 500)
        _render_lidar_image(np.zeros((10, 2), np.float32), np.zeros(10, np.float32), 
                          np.zeros(10, np.float32), 64, 64, 2)
        
        print("  ✓ Calibrator ready")
        print(f"  Press 'o' to inject offset: T={config.trans_offset*100}cm, R={config.rot_offset}deg")
        print(f"  Press 'r' to reset to initial calibration")
    
    def inject_offset(self):
        trans_offset = self.config.trans_offset  # [x, y, z] in meters
        rot_offset = self.config.rot_offset      # [rx, ry, rz] in degrees
        
        # Store the CORRECT calibration (before offset) as target
        self.T_at_injection = self.T.copy()
        
        # Apply translation offset
        self.T[:3, 3] += trans_offset
        
        # Apply rotation offset (convert degrees to radians)
        if np.any(rot_offset != 0):
            R_offset = Rotation.from_euler('xyz', rot_offset, degrees=True).as_matrix()
            self.T[:3, :3] = R_offset @ self.T[:3, :3]
        
        # Update T_initial to the NEW (wrong) position for drift limits
        self.T_initial = self.T.copy()
        
        self.offset_injected = True
        self.offset_injection_frame = self.frame_count
        self.last_dT = np.zeros(3)
        self.last_dR = np.zeros(3)
        
        print(f"\n{'='*60}")
        print(f"  OFFSET INJECTED at frame {self.frame_count}")
        print(f"  Translation offset: [{trans_offset[0]*100:.2f}, {trans_offset[1]*100:.2f}, {trans_offset[2]*100:.2f}] cm")
        print(f"  Rotation offset:    [{rot_offset[0]:.2f}, {rot_offset[1]:.2f}, {rot_offset[2]:.2f}] deg")
        print(f"  Target T: {self.T_at_injection[:3,3]}")
        print(f"  Current T: {self.T[:3,3]}")
        print(f"{'='*60}\n")
    
    def reset_calibration(self):
        self.T = self.config.initial_T.copy()
        self.T_at_injection = None
        self.offset_injected = False
        self.update_count = 0
        self.last_dT = np.zeros(3)
        self.last_dR = np.zeros(3)
        print(f"\n  CALIBRATION RESET\n")
    
    def undistort(self, img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
    
    def extract_lidar_edges(self, xyz, max_pts=3000):
        if len(xyz) < 100:
            return xyz
        if len(xyz) > 8000:
            idx = np.random.choice(len(xyz), 8000, replace=False)
            xyz = xyz[idx]
        
        ranges = np.linalg.norm(xyz, axis=1)
        valid_ranges = np.isfinite(ranges) & (ranges > 0.5) & (ranges < 100)
        xyz = xyz[valid_ranges]
        ranges = ranges[valid_ranges]
        
        if len(xyz) < 100:
            return xyz
        
        tree = cKDTree(xyz)
        _, nn_idx = tree.query(xyz, k=5)
        neighbor_ranges = ranges[nn_idx[:, 1:]]
        max_diff = np.max(np.abs(neighbor_ranges - ranges[:, None]), axis=1)
        
        edge_mask = max_diff > 0.1
        edges = xyz[edge_mask]
        
        if len(edges) > max_pts:
            idx = np.random.choice(len(edges), max_pts, replace=False)
            edges = edges[idx]
        
        return edges
    
    def optimize_transform_grid(self, edges_3d, cam_edges):
        """Grid search with DRIFT LIMITS to prevent runaway."""
        dt = cv2.distanceTransform(255 - cam_edges, cv2.DIST_L2, 5)
        
        T_current = self.T.copy()
        T_init = self.T_initial
        K = self.K
        h, w = self.h, self.w
        edges_3d_f = edges_3d.astype(np.float64)
        
        # Maximum allowed drift from initial calibration
        MAX_T_DRIFT = 0.05  # 5cm max from initial
        MAX_R_DRIFT = 0.05  # ~3 degrees max from initial
        
        def cost_fn(params):
            R_delta = Rotation.from_rotvec(params[:3]).as_matrix()
            R_new = R_delta @ T_current[:3, :3]
            t_new = T_current[:3, 3] + params[3:]
            
            # Check drift from initial calibration
            t_drift = np.linalg.norm(t_new - T_init[:3, 3])
            R_diff = R_new @ T_init[:3, :3].T
            r_drift = np.linalg.norm(Rotation.from_matrix(R_diff).as_rotvec())
            
            # Penalize drift heavily
            if t_drift > MAX_T_DRIFT or r_drift > MAX_R_DRIFT:
                return 1000.0 + t_drift * 100 + r_drift * 100
            
            pts_cam = (R_new @ edges_3d_f.T).T + t_new
            valid = pts_cam[:, 2] > 0.3
            if np.sum(valid) < 50:
                return 1000.0
            
            pts_cam = pts_cam[valid]
            z = pts_cam[:, 2]
            u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
            
            in_bounds = (u >= 0) & (u < w-1) & (v >= 0) & (v < h-1)
            if np.sum(in_bounds) < 50:
                return 1000.0
            
            u_int = u[in_bounds].astype(np.int32)
            v_int = v[in_bounds].astype(np.int32)
            edge_cost = np.mean(dt[v_int, u_int])
            
            # Add small regularization toward initial calibration
            reg = 0.1 * (t_drift * 10 + r_drift * 5)
            return edge_cost + reg
        
        cost_before = cost_fn(np.zeros(6))
        
        best_params = np.zeros(6)
        best_cost = cost_before
        
        # Smaller search range: -2cm to +2cm
        t_range = np.arange(-0.02, 0.021, 0.005)
        r_range = np.arange(-0.015, 0.016, 0.005)
        
        # Grid search - translation only
        for tx in t_range:
            for ty in t_range:
                for tz in t_range:
                    params = np.array([0, 0, 0, tx, ty, tz])
                    c = cost_fn(params)
                    if c < best_cost:
                        best_cost = c
                        best_params = params.copy()
        
        # If translation helped, also search rotation
        if best_cost < cost_before - 0.1:
            t_best = best_params[3:].copy()
            for rx in r_range:
                for ry in r_range:
                    for rz in r_range:
                        params = np.array([rx, ry, rz, t_best[0], t_best[1], t_best[2]])
                        c = cost_fn(params)
                        if c < best_cost:
                            best_cost = c
                            best_params = params.copy()
        
        self.last_opt_result = {
            'cost_before': cost_before,
            'cost_after': best_cost,
            'best_params': best_params
        }
        
        improvement = cost_before - best_cost
        if improvement > 0.1:
            return best_params, cost_before, best_cost, f"Improved by {improvement:.2f}px"
        else:
            return None, cost_before, best_cost, f"No improvement"
    
    def apply_update(self, delta_params):
        r = delta_params[:3].copy()
        t = delta_params[3:].copy()
        
        r_mag = np.linalg.norm(r)
        t_mag = np.linalg.norm(t)
        
        if r_mag > self.config.max_rotation_update:
            r = r * self.config.max_rotation_update / r_mag
        if t_mag > self.config.max_translation_update:
            t = t * self.config.max_translation_update / t_mag
        
        alpha = self.config.update_alpha
        r_smooth = alpha * r
        t_smooth = alpha * t
        
        R_delta = Rotation.from_rotvec(r_smooth).as_matrix()
        T_new = np.eye(4)
        T_new[:3, :3] = R_delta @ self.T[:3, :3]
        T_new[:3, 3] = self.T[:3, 3] + t_smooth
        
        U, _, Vt = np.linalg.svd(T_new[:3, :3])
        T_new[:3, :3] = U @ Vt
        
        T_before = self.T.copy()
        self.T = T_new
        
        self.last_dT = (self.T[:3, 3] - T_before[:3, 3]) * 100
        R_diff = self.T[:3, :3] @ T_before[:3, :3].T
        self.last_dR = Rotation.from_matrix(R_diff).as_euler('xyz', degrees=True)
        
        return self.last_dR, self.last_dT
    
    def generate_lidar_image(self, xyz, intensity):
        R, t = self.T[:3, :3], self.T[:3, 3]
        uv, depths = _project_points_fast(
            xyz.astype(np.float32), R.astype(np.float64), t.astype(np.float64),
            self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2])
        img = _render_lidar_image(uv, depths, intensity.astype(np.float32), self.h, self.w, 2)
        if img.max() > 0:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img.astype(np.uint8)
    
    def process(self, lidar_pts, cam_img):
        self.frame_count += 1
        t0 = time.time()
        
        if cam_img is None:
            return self._empty_result()
        
        cam_rect = self.undistort(cam_img)
        gray = cv2.cvtColor(cam_rect, cv2.COLOR_BGR2GRAY) if len(cam_rect.shape) == 3 else cam_rect
        cam_edges = cv2.Canny(gray, 50, 150)
        
        xyz = lidar_pts[:, :3]
        intensity = lidar_pts[:, 3] if lidar_pts.shape[1] >= 4 else np.linalg.norm(xyz, axis=1)
        edges_3d = self.extract_lidar_edges(xyz)
        
        result = {
            'frame': self.frame_count, 'updated': False,
            'edge_cost': 50.0, 'cost_before': 50.0, 'cost_after': 50.0,
            'T': self.T.copy(), 'opt_status': 'N/A'
        }
        
        if len(edges_3d) >= 100:
            delta, cost_before, cost_after, status = self.optimize_transform_grid(edges_3d, cam_edges)
            
            result['cost_before'] = cost_before
            result['cost_after'] = cost_after
            result['edge_cost'] = cost_after if delta is not None else cost_before
            result['opt_status'] = status
            
            if delta is not None:
                r_deg, t_cm = self.apply_update(delta)
                result['updated'] = True
                self.update_count += 1
                print(f"    -> Update #{self.update_count}: dT=[{t_cm[0]:.3f}, {t_cm[1]:.3f}, {t_cm[2]:.3f}]cm, "
                      f"dR=[{r_deg[0]:.4f}, {r_deg[1]:.4f}, {r_deg[2]:.4f}]° | "
                      f"Cost: {cost_before:.1f} -> {cost_after:.1f}px")
            else:
                if self.offset_injected and self.frame_count % 10 == 0:
                    print(f"    [Frame {self.frame_count}] {status}")
        
        if not result['updated']:
            self.last_dT = np.zeros(3)
            self.last_dR = np.zeros(3)
        
        self.edge_costs.append(result['edge_cost'])
        result['confidence'] = np.exp(-result['edge_cost'] / 15.0)
        result['T'] = self.T.copy()
        result['processing_time_ms'] = (time.time() - t0) * 1000
        
        lidar_img = self.generate_lidar_image(xyz, intensity)
        result['vis'] = self._visualize(cam_rect, lidar_img, xyz, cam_edges, result)
        
        return result
    
    def _empty_result(self):
        return {'frame': self.frame_count, 'updated': False, 'edge_cost': 100.0,
                'confidence': 0, 'T': self.T.copy(), 'processing_time_ms': 0,
                'vis': np.zeros((self.h, self.w, 3), np.uint8), 'opt_status': 'N/A'}
    
    def _visualize(self, cam_img, lidar_img, xyz, cam_edges, result):
        vis = cam_img.copy() if len(cam_img.shape) == 3 else cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        
        xyz_sub = xyz[::5]
        R, t = self.T[:3, :3], self.T[:3, 3]
        uv, depths = _project_points_fast(xyz_sub.astype(np.float32), R.astype(np.float64), t.astype(np.float64),
                                          self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2])
        
        valid = (depths > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv_valid = uv[valid].astype(np.int32)
        depths_valid = depths[valid]
        
        if len(depths_valid) > 0:
            d_min, d_max = np.percentile(depths_valid, [5, 95])
            d_norm = np.clip((depths_valid - d_min) / (d_max - d_min + 1e-6), 0, 1)
            colors = (plt_jet(d_norm) * 255).astype(np.uint8)
            for i in range(0, len(uv_valid), max(1, len(uv_valid) // 3000)):
                cv2.circle(vis, tuple(uv_valid[i]), 2, tuple(map(int, colors[i, :3][::-1])), -1)
        
        edge_overlay = np.zeros_like(vis)
        edge_overlay[:, :, 1] = cam_edges
        vis = cv2.addWeighted(vis, 0.85, edge_overlay, 0.15, 0)
        
        inset_h, inset_w = h // 4, w // 4
        lidar_small = cv2.resize(cv2.cvtColor(lidar_img, cv2.COLOR_GRAY2BGR), (inset_w, inset_h))
        vis[10:10+inset_h, 10:10+inset_w] = lidar_small
        cv2.rectangle(vis, (10, 10), (10+inset_w, 10+inset_h), (255, 255, 255), 2)
        
        edges_small = cv2.resize(cv2.cvtColor(cam_edges, cv2.COLOR_GRAY2BGR), (inset_w, inset_h))
        vis[20+inset_h:20+2*inset_h, 10:10+inset_w] = edges_small
        cv2.rectangle(vis, (10, 20+inset_h), (10+inset_w, 20+2*inset_h), (0, 255, 0), 2)
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (w-400, 5), (w-5, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        
        euler = Rotation.from_matrix(self.T[:3, :3]).as_euler('xyz', degrees=True)
        trans = self.T[:3, 3]
        
        color = (0, 255, 0) if result['updated'] else (255, 255, 255)
        y = 25
        cv2.putText(vis, f"Frame: {result['frame']}", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1); y += 22
        cv2.putText(vis, f"Edge Cost: {result['edge_cost']:.2f}px", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y += 22
        cv2.putText(vis, f"Updates: {self.update_count}", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y += 22
        cv2.putText(vis, f"Time: {result['processing_time_ms']:.0f}ms", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y += 28
        
        cv2.putText(vis, "Current Transform:", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 18
        cv2.putText(vis, f"T: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]m", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1); y += 18
        cv2.putText(vis, f"R: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]deg", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1); y += 24
        
        dt_color = (0, 255, 0) if np.any(np.abs(self.last_dT) > 0.01) else (128, 128, 128)
        cv2.putText(vis, "Last Update:", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1); y += 18
        cv2.putText(vis, f"dT: [{self.last_dT[0]:.3f}, {self.last_dT[1]:.3f}, {self.last_dT[2]:.3f}]cm", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dt_color, 1); y += 18
        cv2.putText(vis, f"dR: [{self.last_dR[0]:.4f}, {self.last_dR[1]:.4f}, {self.last_dR[2]:.4f}]deg", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dt_color, 1); y += 24
        
        if self.offset_injected and self.T_at_injection is not None:
            frames_since = self.frame_count - self.offset_injection_frame
            
            # Translation error
            target_t = self.T_at_injection[:3, 3]
            current_t = self.T[:3, 3]
            t_error = current_t - target_t
            initial_t_error = np.linalg.norm(self.config.trans_offset)
            current_t_error = np.linalg.norm(t_error)
            
            # Rotation error
            R_target = self.T_at_injection[:3, :3]
            R_current = self.T[:3, :3]
            R_error = R_current @ R_target.T
            r_error_vec = Rotation.from_matrix(R_error).as_euler('xyz', degrees=True)
            initial_r_error = np.linalg.norm(self.config.rot_offset)
            current_r_error = np.linalg.norm(r_error_vec)
            
            # Recovery percentages
            t_recovery = 100 * (1 - current_t_error / initial_t_error) if initial_t_error > 0.001 else 100
            r_recovery = 100 * (1 - current_r_error / initial_r_error) if initial_r_error > 0.01 else 100
            
            cv2.putText(vis, f"OFFSET TEST (Frame +{frames_since})", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2); y += 18
            cv2.putText(vis, f"T err: [{t_error[0]*100:.2f}, {t_error[1]*100:.2f}, {t_error[2]*100:.2f}]cm", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1); y += 16
            cv2.putText(vis, f"R err: [{r_error_vec[0]:.2f}, {r_error_vec[1]:.2f}, {r_error_vec[2]:.2f}]deg", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1); y += 16
            
            t_color = (0, 255, 0) if t_recovery > 50 else (0, 165, 255) if t_recovery > 0 else (0, 0, 255)
            r_color = (0, 255, 0) if r_recovery > 50 else (0, 165, 255) if r_recovery > 0 else (0, 0, 255)
            cv2.putText(vis, f"T recovery: {t_recovery:.1f}% | R recovery: {r_recovery:.1f}%", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, t_color, 1)
        else:
            cv2.putText(vis, "Press 'o' to inject offset", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1); y += 18
            cv2.putText(vis, "Press 'r' to reset", (w-390, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
        
        if result['updated']:
            cv2.putText(vis, "UPDATED", (w-100, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
    
    def save(self, path="calibration_online.json"):
        with open(path, 'w') as f:
            json.dump({'T': self.T.tolist(), 'updates': self.update_count}, f, indent=2)
        print(f"Saved to {path}")

def plt_jet(x):
    x = np.clip(x, 0, 1)
    return np.stack([np.clip(1.5 - np.abs(4*x - 3), 0, 1),
                     np.clip(1.5 - np.abs(4*x - 2), 0, 1),
                     np.clip(1.5 - np.abs(4*x - 1), 0, 1)], -1)

class StreamingLoader:
    def __init__(self, path):
        self.files = sorted(Path(path).glob("*.mcap"))
        print(f"Found {len(self.files)} MCAP files")
    
    def stream(self, lidar_topic="/ouster/points", cam_topic="/cam_sync/cam0/image_raw/compressed", sync_thresh=0.05):
        decoder = DecoderFactory()
        lidar_buf, cam_buf = deque(maxlen=5), deque(maxlen=10)
        for f in self.files:
            print(f"\nStreaming: {f.name}")
            try:
                with open(f, 'rb') as fp:
                    reader = make_reader(fp, decoder_factories=[decoder])
                    for schema, channel, msg in reader.iter_messages(topics=[lidar_topic, cam_topic]):
                        try:
                            ros_msg = decoder.decoder_for(channel.message_encoding, schema)(msg.data)
                            ts = msg.log_time / 1e9
                            if channel.topic == lidar_topic:
                                pts = parse_pc2(ros_msg)
                                if pts is not None and len(pts) > 100:
                                    lidar_buf.append((ts, pts))
                                    if cam_buf:
                                        best = min(cam_buf, key=lambda x: abs(x[0] - ts))
                                        if abs(best[0] - ts) < sync_thresh:
                                            yield {'lidar': pts, 'camera': best[1]}
                            elif channel.topic == cam_topic:
                                img = parse_img(ros_msg)
                                if img is not None:
                                    cam_buf.append((ts, img))
                        except (struct.error, ValueError):
                            continue
            except Exception as e:
                print(f"  Error: {e}")

def parse_pc2(msg):
    fields = {f.name: f.offset for f in msg.fields}
    data = msg.data.tobytes() if hasattr(msg.data, 'tobytes') else bytes(msg.data)
    n = msg.width * msg.height
    try:
        names, fmts, offs = ['x', 'y', 'z'], ['<f4', '<f4', '<f4'], [fields.get('x', 0), fields.get('y', 4), fields.get('z', 8)]
        if 'intensity' in fields or 'signal' in fields:
            names.append('intensity'); fmts.append('<f4')
            offs.append(fields.get('intensity', fields.get('signal')))
        dt = np.dtype({'names': names, 'formats': fmts, 'offsets': offs, 'itemsize': msg.point_step})
        s = np.frombuffer(data, dtype=dt, count=n)
        cols = [s['x'], s['y'], s['z']]
        if 'intensity' in names: cols.append(s['intensity'])
        pts = np.column_stack(cols)
        valid = (np.linalg.norm(pts[:, :3], axis=1) > 0.5) & ~np.isnan(pts[:, :3]).any(axis=1)
        return pts[valid].astype(np.float32)
    except:
        return None

def parse_img(msg):
    data = bytes(msg.data) if hasattr(msg.data, '__iter__') else msg.data
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def parse_offset(s):
    """Parse comma-separated offset string like '0.02,0,0' or '2,0,0'."""
    try:
        vals = [float(x.strip()) for x in s.split(',')]
        if len(vals) == 1:
            return np.array([vals[0], 0.0, 0.0])
        elif len(vals) == 3:
            return np.array(vals)
        else:
            print(f"Warning: Expected 1 or 3 values, got {len(vals)}. Using first value only.")
            return np.array([vals[0], 0.0, 0.0])
    except:
        print(f"Warning: Could not parse '{s}'. Using zeros.")
        return np.array([0.0, 0.0, 0.0])

def main():
    import sys
    DATA_PATH = "."
    trans_offset = np.array([0.02, 0.0, 0.0])  # Default 2cm in X
    rot_offset = np.array([0.0, 0.0, 0.0])     # Default no rotation
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--trans-offset' and i + 1 < len(args):
            trans_offset = parse_offset(args[i + 1])
            # Convert from cm to meters if values seem to be in cm (> 0.5 likely means cm)
            if np.any(np.abs(trans_offset) > 0.5):
                print(f"  Note: Converting translation from cm to m")
                trans_offset = trans_offset / 100.0
            i += 2
        elif arg == '--rot-offset' and i + 1 < len(args):
            rot_offset = parse_offset(args[i + 1])
            i += 2
        elif not arg.startswith('-'):
            DATA_PATH = arg
            i += 1
        else:
            i += 1
    
    print("=" * 60)
    print("Online Camera-LiDAR Calibration")
    print("=" * 60)
    print(f"  Translation offset: [{trans_offset[0]*100:.2f}, {trans_offset[1]*100:.2f}, {trans_offset[2]*100:.2f}] cm")
    print(f"  Rotation offset:    [{rot_offset[0]:.2f}, {rot_offset[1]:.2f}, {rot_offset[2]:.2f}] deg")
    print("=" * 60)
    
    config = OnlineCalibConfig()
    config.trans_offset = trans_offset
    config.rot_offset = rot_offset
    
    calibrator = OnlineCalibrator(config)
    loader = StreamingLoader(DATA_PATH)
    
    h, w = config.camera.height, config.camera.width
    out = cv2.VideoWriter('calibration_online.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    
    print("\n  'o' - inject offset | 'r' - reset | 'q' - quit\n")
    
    for i, frame in enumerate(loader.stream()):
        result = calibrator.process(frame['lidar'], frame['camera'])
        vis = result['vis']
        if vis.shape[:2] != (h, w):
            vis = cv2.resize(vis, (w, h))
        out.write(vis)
        
        try:
            cv2.imshow("Online Calibration", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('o'): calibrator.inject_offset()
            elif key == ord('r'): calibrator.reset_calibration()
        except: pass
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1} | Cost: {result['edge_cost']:.1f}px | Updates: {calibrator.update_count} | {result['processing_time_ms']:.0f}ms")
    
    out.release()
    try: cv2.destroyAllWindows()
    except: pass
    calibrator.save()
    print(f"\nFinal T:\n{calibrator.T}")

if __name__ == "__main__":
    main()