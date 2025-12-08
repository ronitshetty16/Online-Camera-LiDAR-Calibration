#!/usr/bin/env python3
"""
TRUE ONLINE Camera-LiDAR Extrinsic Calibration
Edge-based alignment with feature matching support.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Generator
from pathlib import Path
from scipy.spatial.transform import Rotation
from collections import deque
import time, json, yaml, struct

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CameraConfig:
    fx: float = 1467.823; fy: float = 1468.471
    cx: float = 616.833; cy: float = 500.724
    width: int = 1224; height: int = 1024
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.array([-0.2306, 0.207, 0.0005, -0.002]))
    
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

@dataclass
class OnlineCalibConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    initial_T: np.ndarray = field(default_factory=lambda: np.array([
        [0.99996, 0.00576, -0.00691, 0.1049], [-0.00724, 0.05981, -0.99818, 0.1105],
        [-0.00533, 0.99819, 0.05985, -0.1113], [0., 0., 0., 1.]]))
    matcher_backend: str = 'edge'
    min_matches: int = 8; ransac_thresh: float = 5.0
    update_alpha: float = 0.3
    min_inliers_for_update: int = 15
    max_rotation_update: float = 0.05
    max_translation_update: float = 0.02
    confidence_window: int = 10; min_confidence_for_update: float = 0.3
    edge_improve_thresh: float = 0.995
    # Performance settings
    skip_frames: int = 2  # Only process every Nth frame for calibration
    max_edge_points: int = 800  # For edge cost computation only

# ============================================================================
# MATCHERS
# ============================================================================

class EdgeBasedMatcher:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        edges1, edges2 = cv2.Canny(g1, 30, 100), cv2.Canny(g2, 30, 100)
        pts1 = np.column_stack(np.where(edges1 > 0))[:, ::-1]
        if len(pts1) < 10: return np.array([]), np.array([])
        
        # Subsample more aggressively
        if len(pts1) > 500: pts1 = pts1[np.random.choice(len(pts1), 500, replace=False)]
        
        dist_transform = cv2.distanceTransform(255 - edges2, cv2.DIST_L2, 3)  # Smaller mask
        
        # Vectorized matching - no loop!
        h, w = edges2.shape
        valid_mask = (pts1[:, 0] >= 0) & (pts1[:, 0] < w) & (pts1[:, 1] >= 0) & (pts1[:, 1] < h)
        pts1 = pts1[valid_mask]
        
        dists = dist_transform[pts1[:, 1], pts1[:, 0]]
        close_mask = dists < 15  # Only keep points close to edges
        
        pts1_valid = pts1[close_mask]
        # For edge matching, pts2 ≈ pts1 (we're measuring alignment, not displacement)
        pts2_valid = pts1_valid.copy()
        
        return pts1_valid, pts2_valid

def create_matcher(backend):
    print(f"  Creating matcher: {backend}")
    return EdgeBasedMatcher()  # Always use edge-based (fastest)

# ============================================================================
# LIDAR IMAGE GENERATOR
# ============================================================================

class LidarImageGenerator:
    def __init__(self, config):
        self.K, self.h, self.w = config.camera.K, config.camera.height, config.camera.width
        self.max_pts = config.max_lidar_points
    
    def generate(self, points, T, radius=1):
        # Subsample early
        if len(points) > self.max_pts:
            idx = np.random.choice(len(points), self.max_pts, replace=False)
            points = points[idx]
        
        xyz = points[:, :3]
        intensity = points[:, 3] if points.shape[1] >= 4 else np.linalg.norm(xyz, axis=1)
        
        # Vectorized transform
        pts_cam = (T[:3,:3] @ xyz.T).T + T[:3, 3]
        valid = pts_cam[:, 2] > 0.5
        pts_cam, intensity, xyz = pts_cam[valid], intensity[valid], xyz[valid]
        if len(pts_cam) == 0: return np.zeros((self.h, self.w), np.uint8), None, None
        
        # Vectorized projection
        uv = (self.K @ pts_cam.T)
        uv = (uv[:2] / uv[2]).T
        depths = pts_cam[:, 2]
        
        ib = (uv[:,0] >= 0) & (uv[:,0] < self.w) & (uv[:,1] >= 0) & (uv[:,1] < self.h)
        uv, intensity, depths, xyz = uv[ib].astype(np.int32), intensity[ib], depths[ib], xyz[ib]
        
        # Use numpy advanced indexing instead of loop
        img = np.zeros((self.h, self.w), np.float32)
        depth_img = np.full((self.h, self.w), np.inf, np.float32)
        xyz_img = np.zeros((self.h, self.w, 3), np.float32)
        
        # Sort by depth (far to near) and use simple assignment
        order = np.argsort(depths)[::-1]
        for i in order:
            u, v = uv[i]
            if depths[i] < depth_img[v, u]:
                depth_img[v, u] = depths[i]
                img[v, u] = intensity[i]
                xyz_img[v, u] = xyz[i]
        
        # Fast normalization and dilation
        if img.max() > 0: 
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Simple dilation instead of inpainting (much faster)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=2)
        
        self._xyz, self._depth = xyz_img, depth_img
        return img, depth_img, xyz_img
    
    def get_3d_at(self, uv):
        if not hasattr(self, '_xyz'): return np.array([]), np.array([])
        uv = np.clip(uv.astype(np.int32), [0,0], [self.w-1, self.h-1])
        pts = self._xyz[uv[:,1], uv[:,0]]
        valid = self._depth[uv[:,1], uv[:,0]] < np.inf
        return pts, valid

# ============================================================================
# ONLINE CALIBRATOR
# ============================================================================

class TrueOnlineCalibrator:
    def __init__(self, config):
        self.config, self.T = config, config.initial_T.copy()
        self.T_initial = config.initial_T.copy()  # Store initial for comparison
        self.T_prev = self.T.copy()  # Store previous frame's T
        print(f"\nInitializing Online Calibrator...\n  Matcher: {config.matcher_backend}")
        self.matcher = create_matcher(config.matcher_backend)
        self.lidar_gen = LidarImageGenerator(config)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            config.camera.K, config.camera.dist_coeffs, None, config.camera.K,
            (config.camera.width, config.camera.height), cv2.CV_32FC1)
        self.frame_count = self.update_count = 0
        self.conf_hist = deque(maxlen=config.confidence_window)
        self.edge_costs = []
        print("  ✓ Ready")
    
    def undistort(self, img): return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
    
    def edge_cost(self, lidar_pts, cam_edges, T=None):
        if T is None: T = self.T
        xyz = lidar_pts[:, :3]
        
        # Subsample more aggressively
        max_pts = self.config.max_edge_points
        if len(xyz) > max_pts:
            idx = np.random.choice(len(xyz), max_pts, replace=False)
            xyz = xyz[idx]
        
        ranges = np.linalg.norm(xyz, axis=1)
        
        # Faster edge detection: use range gradient
        # Sort by azimuth angle for organized-like processing
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        order = np.argsort(azimuth)
        ranges_sorted = ranges[order]
        
        # Find discontinuities via diff
        range_diff = np.abs(np.diff(ranges_sorted))
        edge_idx = order[:-1][range_diff > 0.3]
        
        if len(edge_idx) < 10: return 1e6
        
        edges_3d = xyz[edge_idx]
        
        # Project to camera
        pts_cam = (T[:3,:3] @ edges_3d.T).T + T[:3, 3]
        valid = pts_cam[:, 2] > 0.5
        pts_cam = pts_cam[valid]
        if len(pts_cam) < 10: return 1e6
        
        uv = self.config.camera.K @ pts_cam.T
        uv = (uv[:2] / uv[2]).T
        
        h, w = cam_edges.shape
        ib = (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
        uv = uv[ib].astype(np.int32)
        if len(uv) < 10: return 1e6
        
        # Precompute distance transform once per frame (cached)
        if not hasattr(self, '_cached_dt') or self._cached_edges_id != id(cam_edges):
            self._cached_dt = cv2.distanceTransform(255 - cam_edges, cv2.DIST_L2, 3)
            self._cached_edges_id = id(cam_edges)
        
        return np.mean(np.minimum(self._cached_dt[uv[:,1], uv[:,0]], 20))
    
    def refine_edges(self, lidar_pts, cam_edges):
        """Fast edge refinement with fewer iterations"""
        best_cost = self.edge_cost(lidar_pts, cam_edges)
        best_T = None
        
        # Simple grid search instead of L-BFGS-B (faster for small adjustments)
        for dr in [-0.005, 0, 0.005]:  # ~0.3 degrees
            for dp in [-0.005, 0, 0.005]:
                for dy in [-0.005, 0, 0.005]:
                    for tx in [-0.003, 0, 0.003]:
                        for ty in [-0.003, 0, 0.003]:
                            for tz in [-0.003, 0, 0.003]:
                                if dr == dp == dy == tx == ty == tz == 0:
                                    continue
                                R = Rotation.from_rotvec([dr, dp, dy]).as_matrix()
                                T_test = self.T.copy()
                                T_test[:3,:3] = R @ self.T[:3,:3]
                                T_test[:3,3] += [tx, ty, tz]
                                
                                cost = self.edge_cost(lidar_pts, cam_edges, T_test)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_T = T_test.copy()
        
        return best_T
    
    def apply_update(self, T_new):
        T_delta = T_new @ np.linalg.inv(self.T)
        r = Rotation.from_matrix(T_delta[:3,:3]).as_rotvec()
        t = T_delta[:3, 3]
        
        # Clip to max update
        r_mag = np.linalg.norm(r)
        t_mag = np.linalg.norm(t)
        if r_mag > self.config.max_rotation_update:
            r = r * self.config.max_rotation_update / r_mag
        if t_mag > self.config.max_translation_update:
            t = t * self.config.max_translation_update / t_mag
        
        # Apply with smoothing
        a = self.config.update_alpha
        self.T[:3,:3] = Rotation.from_rotvec(a * r).as_matrix() @ self.T[:3,:3]
        self.T[:3, 3] += a * t
        
        # Re-orthogonalize
        U, _, Vt = np.linalg.svd(self.T[:3,:3]); self.T[:3,:3] = U @ Vt
        
        # Print update for debugging
        r_deg = np.degrees(r)
        t_cm = t * 100
        print(f"    -> Update applied: dR=[{r_deg[0]:.3f}, {r_deg[1]:.3f}, {r_deg[2]:.3f}]deg, dT=[{t_cm[0]:.2f}, {t_cm[1]:.2f}, {t_cm[2]:.2f}]cm")
        
        return True
    
    def process(self, lidar_pts, cam_img, nearir_img=None):
        self.frame_count += 1; t0 = time.time()
        
        # Use ALL LiDAR points (no subsampling)
        cam_rect = self.undistort(cam_img) if cam_img is not None else None
        
        match_img = nearir_img if nearir_img is not None else cam_rect
        if match_img is None: return self._empty_result()
        if match_img.shape[:2] != (self.config.camera.height, self.config.camera.width):
            match_img = cv2.resize(match_img, (self.config.camera.width, self.config.camera.height))
        
        gray = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY) if len(match_img.shape) == 3 else match_img
        cam_edges = cv2.Canny(gray, 30, 100)
        
        # Generate LiDAR edge image for visualization
        lidar_edge_img = self._create_lidar_edge_image(lidar_pts)
        
        result = {'frame': self.frame_count, 'updated': False, 'n_matches': 0,
                  'n_inliers': 0, 'edge_cost': 0, 'confidence': 0, 'T': self.T.copy()}
        
        # Only run calibration every N frames
        if self.frame_count % self.config.skip_frames == 0:
            cost_before = self.edge_cost(lidar_pts, cam_edges)
            
            # Try edge refinement
            T_refined = self.refine_edges(lidar_pts, cam_edges)
            if T_refined is not None:
                cost_after = self.edge_cost(lidar_pts, cam_edges, T_refined)
                if cost_after < cost_before * self.config.edge_improve_thresh:
                    self.T_prev = self.T.copy()
                    self.apply_update(T_refined)
                    result['updated'] = True
                    self.update_count += 1
            
            result['edge_cost'] = self.edge_cost(lidar_pts, cam_edges)
        else:
            result['edge_cost'] = self.edge_costs[-1] if self.edge_costs else 0
        
        result['confidence'] = np.exp(-result['edge_cost'] / 10.0) if result['edge_cost'] < 1e5 else 0
        self.conf_hist.append(result['confidence'])
        self.edge_costs.append(result['edge_cost'])
        result['T'] = self.T.copy()
        result['processing_time_ms'] = (time.time() - t0) * 1000
        
        # Fast visualization with both edge images
        result['vis'] = self._visualize_fast(cam_rect if cam_rect is not None else match_img,
                                              lidar_pts, cam_edges, lidar_edge_img, result)
        return result
    
    def _create_lidar_edge_image(self, lidar_pts):
        """Create image showing projected LiDAR depth discontinuities (edges)"""
        xyz = lidar_pts[:, :3]
        ranges = np.linalg.norm(xyz, axis=1)
        
        # Find depth discontinuities using azimuth sorting
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        order = np.argsort(azimuth)
        ranges_sorted = ranges[order]
        xyz_sorted = xyz[order]
        
        # Find discontinuities
        range_diff = np.abs(np.diff(ranges_sorted))
        edge_mask = np.zeros(len(xyz), dtype=bool)
        edge_idx = order[:-1][range_diff > 0.3]
        edge_mask[edge_idx] = True
        
        edges_3d = xyz[edge_mask]
        
        # Project to image
        img = np.zeros((self.config.camera.height, self.config.camera.width), dtype=np.uint8)
        
        if len(edges_3d) < 10:
            return img
        
        pts_cam = (self.T[:3,:3] @ edges_3d.T).T + self.T[:3, 3]
        valid = pts_cam[:, 2] > 0.5
        pts_cam = pts_cam[valid]
        
        if len(pts_cam) < 10:
            return img
        
        uv = self.config.camera.K @ pts_cam.T
        uv = (uv[:2] / uv[2]).T
        
        h, w = img.shape
        ib = (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
        uv = uv[ib].astype(np.int32)
        
        # Draw edge points
        for u, v in uv:
            cv2.circle(img, (u, v), 2, 255, -1)
        
        # Dilate slightly for visibility
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        
        return img
    
    def _empty_result(self):
        return {'frame': self.frame_count, 'updated': False, 'n_matches': 0, 'n_inliers': 0,
                'edge_cost': 1e6, 'confidence': 0, 'T': self.T.copy(), 'processing_time_ms': 0,
                'vis': np.zeros((self.config.camera.height, self.config.camera.width, 3), np.uint8)}
    
    def _visualize_fast(self, cam_img, lidar_pts, cam_edges, lidar_edge_img, result):
        """Optimized visualization with edge comparison insets"""
        vis = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR) if len(cam_img.shape) == 2 else cam_img.copy()
        h, w = vis.shape[:2]
        
        # Project ALL LiDAR points
        xyz = lidar_pts[:, :3]
        pts_cam = (self.T[:3,:3] @ xyz.T).T + self.T[:3,3]
        valid = pts_cam[:, 2] > 0.5
        
        if np.any(valid):
            pts_cam_v = pts_cam[valid]
            uv = (self.config.camera.K @ pts_cam_v.T)
            uv = (uv[:2] / uv[2]).T
            depths = pts_cam_v[:, 2]
            
            ib = (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
            uv, depths = uv[ib].astype(np.int32), depths[ib]
            
            if len(depths) > 0:
                d_min, d_max = np.percentile(depths, [5, 95])
                d_norm = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)
                colors = (plt_jet(d_norm) * 255).astype(np.uint8)
                
                # Draw all points (subsample only for very large clouds)
                step = max(1, len(uv) // 8000)
                for i in range(0, len(uv), step):
                    cv2.circle(vis, tuple(uv[i]), 2, tuple(map(int, colors[i,:3][::-1])), -1)
        
        # Overlay camera edges faintly on main image
        vis[:,:,1] = np.maximum(vis[:,:,1], cam_edges // 4)
        
        # === INSETS: Side-by-side edge comparison ===
        inset_h, inset_w = h // 4, w // 4
        inset_y = 10
        
        # Inset 1: LiDAR edges (what we're projecting)
        lidar_edge_color = cv2.cvtColor(lidar_edge_img, cv2.COLOR_GRAY2BGR)
        lidar_edge_color[:,:,2] = lidar_edge_img  # Red channel
        lidar_edge_color[:,:,0] = 0  # No blue
        lidar_edge_color[:,:,1] = 0  # No green
        lidar_edge_small = cv2.resize(lidar_edge_color, (inset_w, inset_h))
        vis[inset_y:inset_y+inset_h, 10:10+inset_w] = lidar_edge_small
        cv2.rectangle(vis, (10, inset_y), (10+inset_w, inset_y+inset_h), (0,0,255), 2)
        cv2.putText(vis, "LiDAR Edges", (15, inset_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        # Inset 2: Camera edges (target to match)
        cam_edge_color = cv2.cvtColor(cam_edges, cv2.COLOR_GRAY2BGR)
        cam_edge_color[:,:,1] = cam_edges  # Green channel
        cam_edge_color[:,:,0] = 0
        cam_edge_color[:,:,2] = 0
        cam_edge_small = cv2.resize(cam_edge_color, (inset_w, inset_h))
        vis[inset_y:inset_y+inset_h, 20+inset_w:20+2*inset_w] = cam_edge_small
        cv2.rectangle(vis, (20+inset_w, inset_y), (20+2*inset_w, inset_y+inset_h), (0,255,0), 2)
        cv2.putText(vis, "Camera Edges", (25+inset_w, inset_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        # Inset 3: Overlay (both edges together to see alignment)
        overlay_img = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_img[:,:,1] = cam_edges  # Green = camera
        overlay_img[:,:,2] = lidar_edge_img  # Red = LiDAR
        # Where they overlap = yellow
        overlay_small = cv2.resize(overlay_img, (inset_w, inset_h))
        vis[inset_y:inset_y+inset_h, 30+2*inset_w:30+3*inset_w] = overlay_small
        cv2.rectangle(vis, (30+2*inset_w, inset_y), (30+3*inset_w, inset_y+inset_h), (0,255,255), 2)
        cv2.putText(vis, "Overlay (R=L,G=C)", (35+2*inset_w, inset_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
        
        # === Status overlay (right side) ===
        overlay = vis.copy()
        cv2.rectangle(overlay, (w-280, 5), (w-5, 175), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
        
        euler = Rotation.from_matrix(self.T[:3,:3]).as_euler('xyz', degrees=True)
        t = self.T[:3, 3]
        euler_init = Rotation.from_matrix(self.T_initial[:3,:3]).as_euler('xyz', degrees=True)
        t_init = self.T_initial[:3, 3]
        delta_t = (t - t_init) * 100  # cm
        delta_r = euler - euler_init
        
        color = (0,255,0) if result['updated'] else (255,255,255)
        y = 22
        cv2.putText(vis, f"Frame:{result['frame']} Cost:{result['edge_cost']:.1f}px", 
                   (w-275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1); y += 20
        cv2.putText(vis, f"Updates: {self.update_count}  |  {result['processing_time_ms']:.0f}ms", 
                   (w-275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1); y += 22
        
        cv2.putText(vis, f"T:[{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]", 
                   (w-275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1); y += 18
        cv2.putText(vis, f"R:[{euler[0]:+.2f}, {euler[1]:+.2f}, {euler[2]:+.2f}]", 
                   (w-275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1); y += 22
        
        cv2.putText(vis, "Delta from initial:", (w-275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1); y += 18
        cv2.putText(vis, f"dT:[{delta_t[0]:+.2f}, {delta_t[1]:+.2f}, {delta_t[2]:+.2f}]cm", 
                   (w-275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1); y += 18
        cv2.putText(vis, f"dR:[{delta_r[0]:+.3f}, {delta_r[1]:+.3f}, {delta_r[2]:+.3f}]deg", 
                   (w-275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        
        if result['updated']:
            cv2.putText(vis, "UPD", (w-50, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        return vis
    
    def save(self, path="calibration_online.json"):
        result = {'T': self.T.tolist(), 't': self.T[:3,3].tolist(),
                  'r_deg': Rotation.from_matrix(self.T[:3,:3]).as_euler('xyz', degrees=True).tolist(),
                  'frames': self.frame_count, 'updates': self.update_count}
        with open(path, 'w') as f: json.dump(result, f, indent=2)
        print(f"Saved to {path}")

def plt_jet(x):
    x = np.clip(x, 0, 1)
    return np.stack([np.clip(1.5-np.abs(4*x-3),0,1), np.clip(1.5-np.abs(4*x-2),0,1), np.clip(1.5-np.abs(4*x-1),0,1)], -1)

# ============================================================================
# STREAMING LOADER
# ============================================================================

class StreamingLoader:
    def __init__(self, path):
        self.files = sorted(Path(path).glob("*.mcap"))
        print(f"Found {len(self.files)} MCAP files")
    
    def stream(self, lidar_topic="/ouster/points", cam_topic="/cam_sync/cam0/image_raw/compressed",
               nearir_topic="/ouster/nearir_image", sync_thresh=0.05):
        decoder = DecoderFactory()
        lidar_buf, cam_buf, ir_buf = deque(maxlen=5), deque(maxlen=10), deque(maxlen=10)
        topics = [lidar_topic, cam_topic, nearir_topic]
        
        for f in self.files:
            print(f"\nStreaming: {f.name}")
            try:
                with open(f, 'rb') as fp:
                    reader = make_reader(fp, decoder_factories=[decoder])
                    for schema, channel, msg in reader.iter_messages(topics=topics):
                        try:
                            ros_msg = decoder.decoder_for(channel.message_encoding, schema)(msg.data)
                            ts = msg.log_time / 1e9
                            
                            if channel.topic == lidar_topic:
                                pts = parse_pc2(ros_msg)
                                if pts is not None and len(pts) > 100:
                                    lidar_buf.append((ts, pts))
                                    frame = self._sync(lidar_buf, cam_buf, ir_buf, sync_thresh)
                                    if frame: yield frame
                            elif channel.topic == cam_topic:
                                img = parse_img(ros_msg)
                                if img is not None: cam_buf.append((ts, img))
                            elif channel.topic == nearir_topic:
                                img = parse_ir(ros_msg)
                                if img is not None: ir_buf.append((ts, img))
                        except (struct.error, ValueError): continue
            except Exception as e: print(f"  Error: {e}")
    
    def _sync(self, lidar_buf, cam_buf, ir_buf, thresh):
        if not lidar_buf: return None
        lt, lpts = lidar_buf[-1]
        cam = min(((abs(t-lt), img) for t, img in cam_buf), key=lambda x: x[0], default=(thresh+1, None))
        ir = min(((abs(t-lt), img) for t, img in ir_buf), key=lambda x: x[0], default=(thresh+1, None))
        if cam[0] > thresh and ir[0] > thresh: return None
        return {'lidar': lpts, 'camera': cam[1] if cam[0] <= thresh else None, 'nearir': ir[1] if ir[0] <= thresh else None}

def parse_pc2(msg):
    fields = {f.name: f.offset for f in msg.fields}
    data = msg.data.tobytes() if hasattr(msg.data, 'tobytes') else bytes(msg.data)
    n = msg.width * msg.height
    try:
        names, fmts, offs = ['x','y','z'], ['<f4']*3, [fields.get('x',0), fields.get('y',4), fields.get('z',8)]
        if 'intensity' in fields or 'signal' in fields:
            names.append('intensity'); fmts.append('<f4'); offs.append(fields.get('intensity', fields.get('signal')))
        dt = np.dtype({'names': names, 'formats': fmts, 'offsets': offs, 'itemsize': msg.point_step})
        s = np.frombuffer(data, dtype=dt, count=n)
        pts = np.column_stack([s['x'], s['y'], s['z']] + ([s['intensity']] if 'intensity' in names else []))
        valid = (np.linalg.norm(pts[:,:3], axis=1) > 0.5) & ~np.isnan(pts[:,:3]).any(axis=1)
        return pts[valid].astype(np.float32)
    except: return None

def parse_img(msg):
    data = bytes(msg.data) if hasattr(msg.data, '__iter__') else msg.data
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def parse_ir(msg):
    try:
        if hasattr(msg, 'format'):
            return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        data = msg.data.tobytes() if hasattr(msg.data, 'tobytes') else bytes(msg.data)
        dtype = np.uint16 if '16' in getattr(msg, 'encoding', 'mono16') else np.uint8
        img = np.frombuffer(data, dtype=dtype).reshape(msg.height, msg.width)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if dtype != np.uint8 else img
    except: return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys
    DATA_PATH = "."
    for arg in sys.argv[1:]:
        if not arg.startswith('-'): DATA_PATH = arg
    
    print("="*60 + "\nTRUE ONLINE Camera-LiDAR Calibration\n" + "="*60)
    
    config = OnlineCalibConfig()
    calibrator = TrueOnlineCalibrator(config)
    loader = StreamingLoader(DATA_PATH)
    
    h, w = config.camera.height, config.camera.width
    out = cv2.VideoWriter('calibration_online.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    
    print("\nStarting... Press 'q' to stop\n")
    
    for i, frame in enumerate(loader.stream()):
        result = calibrator.process(frame['lidar'], frame.get('camera'), frame.get('nearir'))
        vis = result['vis']
        if vis.shape[:2] != (h, w): vis = cv2.resize(vis, (w, h))
        out.write(vis)
        
        try:
            cv2.imshow("Online Calibration", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        except: pass
        
        if (i+1) % 10 == 0:
            print(f"  Frame {i+1} | EdgeCost: {result['edge_cost']:.1f}px | Conf: {result['confidence']:.2f} | Updates: {calibrator.update_count} | {result['processing_time_ms']:.0f}ms")
    
    out.release()
    try: cv2.destroyAllWindows()
    except: pass
    
    calibrator.save()
    print(f"\nFinal T:\n{calibrator.T}")

if __name__ == "__main__": main()