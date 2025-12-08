#!/usr/bin/env python3
"""
Online Camera-LiDAR Extrinsic Calibration with Metric Tracking & Plotting
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from collections import deque
import time, json, struct
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

@dataclass
class CameraConfig:
    fx: float = 1501.9374712879626; fy: float = 1498.8879775647906
    cx: float = 566.5690420612353; cy: float = 537.1294320963829
    width: int = 1224; height: int = 1024
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.array([-0.2306, 0.207, 0.0005, -0.002]))
    @property
    def K(self): return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]], dtype=np.float64)

@dataclass
class OnlineCalibConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    initial_T: np.ndarray = field(default_factory=lambda: np.array([
        [0.99919851,0.04002921,0.0,0.15],[0.0,0.0,-1.0,-0.2815789473684212],
        [-0.04002921,0.99919851,0.0,-0.13157894736842124],[0.0,0.0,0.0,1.0]], dtype=float))
    update_alpha: float = 0.5
    max_rotation_update: float = 0.02
    max_translation_update: float = 0.01
    trans_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rot_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

class CalibrationMetrics:
    """Track calibration metrics over time for plotting."""
    def __init__(self):
        self.frames, self.edge_costs, self.cost_before, self.cost_after = [], [], [], []
        self.trans_x, self.trans_y, self.trans_z = [], [], []
        self.rot_x, self.rot_y, self.rot_z = [], [], []
        self.delta_t, self.delta_r = [], []
        self.trans_err_x, self.trans_err_y, self.trans_err_z, self.trans_err_norm = [], [], [], []
        self.rot_err_x, self.rot_err_y, self.rot_err_z, self.rot_err_norm = [], [], [], []
        self.trans_recovery, self.rot_recovery = [], []
        self.proc_times, self.confidence = [], []
        self.offset_frame = None
        self.init_t_offset, self.init_r_offset = None, None
        
    def record(self, frame, result, calib):
        self.frames.append(frame)
        self.edge_costs.append(result.get('edge_cost', 50.0))
        self.cost_before.append(result.get('cost_before', 50.0))
        self.cost_after.append(result.get('cost_after', 50.0))
        T = result['T']
        t = T[:3, 3]
        euler = Rotation.from_matrix(T[:3,:3]).as_euler('xyz', degrees=True)
        self.trans_x.append(t[0]*100); self.trans_y.append(t[1]*100); self.trans_z.append(t[2]*100)
        self.rot_x.append(euler[0]); self.rot_y.append(euler[1]); self.rot_z.append(euler[2])
        self.delta_t.append(np.linalg.norm(calib.last_dT))
        self.delta_r.append(np.linalg.norm(calib.last_dR))
        self.proc_times.append(result.get('processing_time_ms', 0))
        self.confidence.append(result.get('confidence', 0))
        
        if calib.offset_injected and calib.T_at_injection is not None:
            tgt_t, cur_t = calib.T_at_injection[:3,3], T[:3,3]
            t_err = (cur_t - tgt_t) * 100
            self.trans_err_x.append(t_err[0]); self.trans_err_y.append(t_err[1]); self.trans_err_z.append(t_err[2])
            self.trans_err_norm.append(np.linalg.norm(t_err))
            R_err = T[:3,:3] @ calib.T_at_injection[:3,:3].T
            r_err = Rotation.from_matrix(R_err).as_euler('xyz', degrees=True)
            self.rot_err_x.append(r_err[0]); self.rot_err_y.append(r_err[1]); self.rot_err_z.append(r_err[2])
            self.rot_err_norm.append(np.linalg.norm(r_err))
            init_t = np.linalg.norm(self.init_t_offset)*100 if self.init_t_offset is not None else 1
            init_r = np.linalg.norm(self.init_r_offset) if self.init_r_offset is not None else 1
            self.trans_recovery.append(max(0, 100*(1 - np.linalg.norm(t_err)/max(init_t,0.01))))
            self.rot_recovery.append(max(0, 100*(1 - np.linalg.norm(r_err)/max(init_r,0.01))))
        else:
            for lst in [self.trans_err_x, self.trans_err_y, self.trans_err_z, self.trans_err_norm,
                       self.rot_err_x, self.rot_err_y, self.rot_err_z, self.rot_err_norm]: lst.append(0)
            self.trans_recovery.append(100); self.rot_recovery.append(100)
    
    def mark_offset(self, frame, t_off, r_off):
        self.offset_frame = frame
        self.init_t_offset, self.init_r_offset = t_off.copy(), r_off.copy()
    
    def generate_plots(self, path="calibration_metrics.png"):
        if len(self.frames) < 2: print("Not enough data"); return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        f = np.array(self.frames)
        c = {'x':'#e74c3c','y':'#2ecc71','z':'#3498db','n':'#9b59b6'}
        
        def add_offset_line(ax):
            if self.offset_frame: ax.axvline(self.offset_frame, color='r', ls='--', lw=2, label='Offset Injected')
        
        # 1. Edge Cost
        ax = axes[0,0]
        ax.plot(f, self.edge_costs, 'b-', lw=1.5, label='Edge Cost')
        ax.fill_between(f, self.edge_costs, alpha=0.3)
        add_offset_line(ax)
        ax.set_xlabel('Frame'); ax.set_ylabel('Edge Cost (pixels)')
        ax.set_title('Edge Alignment Cost', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # 2. Translation Error
        ax = axes[0,1]
        if self.offset_frame:
            ax.plot(f, self.trans_err_x, color=c['x'], lw=1.5, label='X error')
            ax.plot(f, self.trans_err_y, color=c['y'], lw=1.5, label='Y error')
            ax.plot(f, self.trans_err_z, color=c['z'], lw=1.5, label='Z error')
            ax.plot(f, self.trans_err_norm, color=c['n'], lw=2.5, ls='--', label='Norm')
            ax.axhline(0, color='k', lw=0.5)
            add_offset_line(ax)
        ax.set_xlabel('Frame'); ax.set_ylabel('Translation Error (cm)')
        ax.set_title('Translation Error from Target', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # 3. Rotation Error
        ax = axes[1,0]
        if self.offset_frame:
            ax.plot(f, self.rot_err_x, color=c['x'], lw=1.5, label='Roll error')
            ax.plot(f, self.rot_err_y, color=c['y'], lw=1.5, label='Pitch error')
            ax.plot(f, self.rot_err_z, color=c['z'], lw=1.5, label='Yaw error')
            ax.plot(f, self.rot_err_norm, color=c['n'], lw=2.5, ls='--', label='Norm')
            ax.axhline(0, color='k', lw=0.5)
            add_offset_line(ax)
        ax.set_xlabel('Frame'); ax.set_ylabel('Rotation Error (degrees)')
        ax.set_title('Rotation Error from Target', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # 4. Recovery %
        ax = axes[1,1]
        if self.offset_frame:
            ax.plot(f, self.trans_recovery, color=c['z'], lw=2, label='Translation Recovery')
            ax.plot(f, self.rot_recovery, color=c['x'], lw=2, label='Rotation Recovery')
            ax.axhline(100, color='g', ls='--', lw=1, alpha=0.7, label='100% Recovery')
            ax.axhline(50, color='orange', ls='--', lw=1, alpha=0.7)
            add_offset_line(ax)
            ax.fill_between(f, self.trans_recovery, alpha=0.2, color=c['z'])
            ax.fill_between(f, self.rot_recovery, alpha=0.2, color=c['x'])
        ax.set_xlabel('Frame'); ax.set_ylabel('Recovery (%)')
        ax.set_title('Calibration Recovery Progress', fontsize=12, fontweight='bold')
        ax.set_ylim([-10, 110]); ax.legend(); ax.grid(True, alpha=0.3)
        
        plt.suptitle('Camera-LiDAR Online Calibration Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Plots saved: {path}")
        plt.close('all')
    
    def save_data(self, path="calibration_metrics.json"):
        data = {'frames':self.frames, 'edge_costs':self.edge_costs, 'cost_before':self.cost_before,
                'cost_after':self.cost_after, 'trans':{'x':self.trans_x,'y':self.trans_y,'z':self.trans_z},
                'rot':{'x':self.rot_x,'y':self.rot_y,'z':self.rot_z},
                'trans_err':{'x':self.trans_err_x,'y':self.trans_err_y,'z':self.trans_err_z,'norm':self.trans_err_norm},
                'rot_err':{'x':self.rot_err_x,'y':self.rot_err_y,'z':self.rot_err_z,'norm':self.rot_err_norm},
                'recovery':{'trans':self.trans_recovery,'rot':self.rot_recovery},
                'proc_times':self.proc_times, 'confidence':self.confidence, 'offset_frame':self.offset_frame}
        with open(path,'w') as f: json.dump(data, f, indent=2)
        print(f"  Data saved: {path}")

@njit(cache=True)
def _project_points_fast(xyz, R, t, fx, fy, cx, cy):
    n = len(xyz); uv = np.empty((n,2), dtype=np.float32); depths = np.empty(n, dtype=np.float32)
    for i in range(n):
        px = R[0,0]*xyz[i,0]+R[0,1]*xyz[i,1]+R[0,2]*xyz[i,2]+t[0]
        py = R[1,0]*xyz[i,0]+R[1,1]*xyz[i,1]+R[1,2]*xyz[i,2]+t[1]
        pz = R[2,0]*xyz[i,0]+R[2,1]*xyz[i,1]+R[2,2]*xyz[i,2]+t[2]
        depths[i] = pz
        if pz > 0.3: uv[i,0] = fx*px/pz+cx; uv[i,1] = fy*py/pz+cy
        else: uv[i,0] = -1; uv[i,1] = -1
    return uv, depths

@njit(parallel=True, cache=True)
def _render_lidar_image(uv, depths, intensity, h, w, radius=2):
    img = np.zeros((h,w), dtype=np.float32); depth_buf = np.full((h,w), 1e10, dtype=np.float32)
    for i in prange(len(uv)):
        u,v,d = int(uv[i,0]), int(uv[i,1]), depths[i]
        if d <= 0.3 or u < 0 or v < 0 or u >= w or v >= h: continue
        for du in range(-radius, radius+1):
            for dv in range(-radius, radius+1):
                nu, nv = u+du, v+dv
                if 0 <= nu < w and 0 <= nv < h and d < depth_buf[nv,nu]:
                    depth_buf[nv,nu] = d; img[nv,nu] = intensity[i]
    return img

def plt_jet(x):
    x = np.clip(x, 0, 1)
    return np.stack([np.clip(1.5-np.abs(4*x-3),0,1), np.clip(1.5-np.abs(4*x-2),0,1), np.clip(1.5-np.abs(4*x-1),0,1)], -1)

class OnlineCalibrator:
    def __init__(self, config):
        self.config = config; self.T = config.initial_T.copy(); self.T_initial = config.initial_T.copy()
        self.K = config.camera.K; self.h, self.w = config.camera.height, config.camera.width
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, config.camera.dist_coeffs, None, self.K, (self.w,self.h), cv2.CV_32FC1)
        self.frame_count = self.update_count = 0; self.edge_costs = deque(maxlen=100)
        self.offset_injected = False; self.offset_injection_frame = -1; self.T_at_injection = None
        self.last_dT = np.zeros(3); self.last_dR = np.zeros(3); self.metrics = CalibrationMetrics()
        # Warmup
        dummy = np.zeros((10,3), dtype=np.float32)
        _project_points_fast(dummy, np.eye(3), np.zeros(3), 1000, 1000, 500, 500)
        _render_lidar_image(np.zeros((10,2),np.float32), np.zeros(10,np.float32), np.zeros(10,np.float32), 64, 64, 2)
        print("  ✓ Calibrator ready\n  'o'-offset | 'r'-reset | 'p'-plots | 'q'-quit")
    
    def inject_offset(self):
        self.T_at_injection = self.T.copy()
        self.T[:3,3] += self.config.trans_offset
        if np.any(self.config.rot_offset != 0):
            R_off = Rotation.from_euler('xyz', self.config.rot_offset, degrees=True).as_matrix()
            self.T[:3,:3] = R_off @ self.T[:3,:3]
        self.T_initial = self.T.copy(); self.offset_injected = True; self.offset_injection_frame = self.frame_count
        self.last_dT = np.zeros(3); self.last_dR = np.zeros(3)
        self.metrics.mark_offset(self.frame_count, self.config.trans_offset, self.config.rot_offset)
        print(f"\n{'='*50}\n  OFFSET INJECTED at frame {self.frame_count}")
        print(f"  T: {self.config.trans_offset*100}cm | R: {self.config.rot_offset}deg\n{'='*50}\n")
    
    def reset_calibration(self):
        self.T = self.config.initial_T.copy(); self.T_at_injection = None; self.offset_injected = False
        self.update_count = 0; self.last_dT = np.zeros(3); self.last_dR = np.zeros(3)
        print("  CALIBRATION RESET")
    
    def undistort(self, img): return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
    
    def extract_lidar_edges(self, xyz, max_pts=8000):
        """    
        For each point, compute the maximum range difference to its 4 nearest neighbors (excluding itself at index 0).

        Point A at 5m range, neighbors at 5.02m, 5.01m, 4.98m, 5.03m → max_diff = 0.05m (NOT an edge)
        Point B at 5m range, neighbors at 5.01m, 5.02m, 8.5m, 8.6m → max_diff = 3.6m (IS an edge)       
        """    
        
        if len(xyz) < 100: return xyz
        if len(xyz) > 8000: xyz = xyz[np.random.choice(len(xyz), 8000, replace=False)]
        ranges = np.linalg.norm(xyz, axis=1)
        valid = np.isfinite(ranges) & (ranges > 0.5) & (ranges < 100)
        xyz, ranges = xyz[valid], ranges[valid]
        if len(xyz) < 100: return xyz
        tree = cKDTree(xyz); _, nn_idx = tree.query(xyz, k=5)
        max_diff = np.max(np.abs(ranges[nn_idx[:,1:]] - ranges[:,None]), axis=1)
        edges = xyz[max_diff > 0.1]
        if len(edges) > max_pts: edges = edges[np.random.choice(len(edges), max_pts, replace=False)]
        return edges
    
    def optimize_transform_grid(self, edges_3d, cam_edges):
        # **Distance Transform:** Creates an image where each pixel value = distance to nearest edge.
        dt = cv2.distanceTransform(255-cam_edges, cv2.DIST_L2, 5)
        T_curr, T_init, K, h, w = self.T.copy(), self.T_initial, self.K, self.h, self.w
        edges = edges_3d.astype(np.float64)
        MAX_T, MAX_R = 0.05, 0.05
        def cost(params):
            # params is a 6-element array: [rx, ry, rz, tx, ty, tz]
            R_d = Rotation.from_rotvec(params[:3]).as_matrix()
            
            # Apply delta transform to current transform
            R_new, t_new = R_d @ T_curr[:3,:3], T_curr[:3,3] + params[3:]
            
            # Calculate how far translation has drifted from initial calibration
            # This is Euclidean distance: sqrt(dx² + dy² + dz²) in meters
            t_drift = np.linalg.norm(t_new - T_init[:3,3])
            
            # Calculate how far rotation has drifted from initial calibration
            r_drift = np.linalg.norm(Rotation.from_matrix(R_new @ T_init[:3,:3].T).as_rotvec())
            
            # Return huge cost (1000+) to ensure optimizer never picks this
            # Extra penalty proportional to drift helps gradient if using continuous optimizer
            if t_drift > MAX_T or r_drift > MAX_R: 
                return 1000.0 + t_drift*100 + r_drift*100
            
            # Transform LiDAR edge points from LiDAR frame to camera frame
            # For each point p: p_camera = R_new @ p_lidar + t_new
            # edges.T is 3xN, R_new @ edges.T is 3xN, transpose back to Nx3, then add translation
            pts = (R_new @ edges.T).T + t_new
            
            # Filter out points that are behind or too close to camera
            # Z > 0.3m means point is at least 30cm in front of camera
            # Points with Z <= 0 are behind camera and can't be projected
            valid = pts[:,2] > 0.3
            
            # Need minimum 50 valid points for reliable cost estimation
            # Too few points = unreliable cost, reject this candidate
            if np.sum(valid) < 50: 
                return 1000.0
            
            # Keep only valid points and extract their Z coordinates (depth)
            pts = pts[valid]
            z = pts[:,2]
            
            # PINHOLE CAMERA PROJECTION: 3D point (X,Y,Z) → 2D pixel (u,v)
            u, v = K[0,0]*pts[:,0]/z + K[0,2], K[1,1]*pts[:,1]/z + K[1,2]
            
            # Filter points that project outside image boundaries
            # Image is w=1224 wide, h=1024 tall
            # Using w-1 and h-1 to avoid edge indexing issues
            inb = (u >= 0) & (u < w-1) & (v >= 0) & (v < h-1)
            
            # Again need minimum 50 in-bounds points for reliable cost
            if np.sum(inb) < 50: 
                return 1000.0
            
            # CORE COST CALCULATION:
            # dt is the distance transform image where each pixel value = 
            # Euclidean distance to nearest camera edge (in pixels)
            # 
            # dt[v, u] samples the distance transform at projected LiDAR edge locations
            # v[inb] and u[inb] are the in-bounds pixel coordinates (as integers for indexing)
            # 
            # np.mean(...) = average distance from projected LiDAR edges to camera edges
            # 
            # GOOD calibration: LiDAR edges project onto camera edges → dt values ≈ 0 → low cost
            # BAD calibration: LiDAR edges project away from camera edges → dt values high → high cost
            #
            # REGULARIZATION TERM: 0.1*(t_drift*10 + r_drift*5)
            # Small penalty for drifting from initial calibration
            # Helps break ties when multiple transforms give similar edge alignment
            # Translation penalized more (×10) than rotation (×5) because translation
            # errors are more visually obvious
            return np.mean(dt[v[inb].astype(np.int32), u[inb].astype(np.int32)]) + 0.1*(t_drift*10 + r_drift*5)
       ### Grid Search ### 
        cost_before = cost(np.zeros(6));
        best_p, best_c = np.zeros(6), cost_before
        for tx in np.arange(-0.02, 0.021, 0.005):
            for ty in np.arange(-0.02, 0.021, 0.005):
                for tz in np.arange(-0.02, 0.021, 0.005):
                    c = cost(np.array([0,0,0,tx,ty,tz]))
                    if c < best_c: best_c, best_p = c, np.array([0,0,0,tx,ty,tz])
        if best_c < cost_before - 0.1:
            tb = best_p[3:].copy()
            for rx in np.arange(-0.015, 0.016, 0.005):
                for ry in np.arange(-0.015, 0.016, 0.005):
                    for rz in np.arange(-0.015, 0.016, 0.005):
                        c = cost(np.array([rx,ry,rz,tb[0],tb[1],tb[2]]))
                        if c < best_c: best_c, best_p = c, np.array([rx,ry,rz,tb[0],tb[1],tb[2]])
        imp = cost_before - best_c
        if imp > 0.1: return best_p, cost_before, best_c, f"Improved {imp:.2f}px"
        return None, cost_before, best_c, "No improvement"
    
    
    def apply_update(self, delta):
        r, t = delta[:3].copy(), delta[3:].copy()
        r_m, t_m = np.linalg.norm(r), np.linalg.norm(t)
        if r_m > self.config.max_rotation_update: r *= self.config.max_rotation_update/r_m
        if t_m > self.config.max_translation_update: t *= self.config.max_translation_update/t_m
        a = self.config.update_alpha
        T_new = np.eye(4)
        T_new[:3,:3] = Rotation.from_rotvec(a*r).as_matrix() @ self.T[:3,:3]
        T_new[:3,3] = self.T[:3,3] + a*t
        U, _, Vt = np.linalg.svd(T_new[:3,:3]); T_new[:3,:3] = U @ Vt
        T_before = self.T.copy(); self.T = T_new
        self.last_dT = (self.T[:3,3] - T_before[:3,3]) * 100
        self.last_dR = Rotation.from_matrix(self.T[:3,:3] @ T_before[:3,:3].T).as_euler('xyz', degrees=True)
        return self.last_dR, self.last_dT
    
    def generate_lidar_image(self, xyz, intensity):
        R, t = self.T[:3,:3], self.T[:3,3]
        uv, depths = _project_points_fast(xyz.astype(np.float32), R.astype(np.float64), t.astype(np.float64), self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2])
        img = _render_lidar_image(uv, depths, intensity.astype(np.float32), self.h, self.w, 2)
        if img.max() > 0: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img.astype(np.uint8)
    
    def process(self, lidar_pts, cam_img):
        self.frame_count += 1; t0 = time.time()
        if cam_img is None: return self._empty_result()
        cam_rect = self.undistort(cam_img)
        gray = cv2.cvtColor(cam_rect, cv2.COLOR_BGR2GRAY) if len(cam_rect.shape)==3 else cam_rect
        cam_edges = cv2.Canny(gray, 50, 150)
        xyz = lidar_pts[:,:3]
        intensity = lidar_pts[:,3] if lidar_pts.shape[1]>=4 else np.linalg.norm(xyz, axis=1)
        edges_3d = self.extract_lidar_edges(xyz)
        result = {'frame':self.frame_count, 'updated':False, 'edge_cost':50.0, 'cost_before':50.0, 'cost_after':50.0, 'T':self.T.copy(), 'opt_status':'N/A'}
        if len(edges_3d) >= 100:
            delta, cb, ca, status = self.optimize_transform_grid(edges_3d, cam_edges)
            result['cost_before'], result['cost_after'] = cb, ca
            result['edge_cost'] = ca if delta is not None else cb
            result['opt_status'] = status
            if delta is not None:
                r_deg, t_cm = self.apply_update(delta)
                result['updated'] = True; self.update_count += 1
                print(f"    Update #{self.update_count}: dT={t_cm}cm dR={r_deg}° | {cb:.1f}->{ca:.1f}px")
        if not result['updated']: self.last_dT = np.zeros(3); self.last_dR = np.zeros(3)
        self.edge_costs.append(result['edge_cost'])
        result['confidence'] = np.exp(-result['edge_cost']/15.0)
        result['T'] = self.T.copy()
        result['processing_time_ms'] = (time.time()-t0)*1000
        self.metrics.record(self.frame_count, result, self)
        result['vis'] = self._visualize(cam_rect, self.generate_lidar_image(xyz, intensity), xyz, cam_edges, result)
        return result
    
    def _empty_result(self):
        return {'frame':self.frame_count, 'updated':False, 'edge_cost':100.0, 'confidence':0, 'T':self.T.copy(), 'processing_time_ms':0, 'vis':np.zeros((self.h,self.w,3),np.uint8), 'opt_status':'N/A'}
    
    def _visualize(self, cam_img, lidar_img, xyz, cam_edges, result):
        vis = cam_img.copy() if len(cam_img.shape)==3 else cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]; xyz_sub = xyz[::5]
        uv, depths = _project_points_fast(xyz_sub.astype(np.float32), self.T[:3,:3].astype(np.float64), self.T[:3,3].astype(np.float64), self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2])
        valid = (depths > 0.3) & (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
        uv_v, d_v = uv[valid].astype(np.int32), depths[valid]
        if len(d_v) > 0:
            d_min, d_max = np.percentile(d_v, [5,95])
            colors = (plt_jet(np.clip((d_v-d_min)/(d_max-d_min+1e-6), 0, 1))*255).astype(np.uint8)
            for i in range(0, len(uv_v), max(1, len(uv_v)//3000)):
                cv2.circle(vis, tuple(uv_v[i]), 2, tuple(map(int, colors[i,:3][::-1])), -1)
        edge_ov = np.zeros_like(vis); edge_ov[:,:,1] = cam_edges
        vis = cv2.addWeighted(vis, 0.85, edge_ov, 0.15, 0)
        ih, iw = h//4, w//4
        vis[10:10+ih, 10:10+iw] = cv2.resize(cv2.cvtColor(lidar_img, cv2.COLOR_GRAY2BGR), (iw,ih))
        cv2.rectangle(vis, (10,10), (10+iw,10+ih), (255,255,255), 2)
        vis[20+ih:20+2*ih, 10:10+iw] = cv2.resize(cv2.cvtColor(cam_edges, cv2.COLOR_GRAY2BGR), (iw,ih))
        cv2.rectangle(vis, (10,20+ih), (10+iw,20+2*ih), (0,255,0), 2)
        ov = vis.copy(); cv2.rectangle(ov, (w-400,5), (w-5,320), (0,0,0), -1)
        cv2.addWeighted(ov, 0.6, vis, 0.4, 0, vis)
        euler = Rotation.from_matrix(self.T[:3,:3]).as_euler('xyz', degrees=True)
        trans = self.T[:3,3]; col = (0,255,0) if result['updated'] else (255,255,255); y = 25
        for txt in [f"Frame: {result['frame']}", f"Edge Cost: {result['edge_cost']:.2f}px", f"Updates: {self.update_count}", f"Time: {result['processing_time_ms']:.0f}ms"]:
            cv2.putText(vis, txt, (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if 'Frame' in txt else (255,255,255), 1); y += 22
        y += 6
        cv2.putText(vis, "Current Transform:", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 18
        cv2.putText(vis, f"T: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]m", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1); y += 18
        cv2.putText(vis, f"R: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]deg", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1); y += 24
        dc = (0,255,0) if np.any(np.abs(self.last_dT) > 0.01) else (128,128,128)
        cv2.putText(vis, "Last Update:", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1); y += 18
        cv2.putText(vis, f"dT: [{self.last_dT[0]:.3f}, {self.last_dT[1]:.3f}, {self.last_dT[2]:.3f}]cm", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dc, 1); y += 18
        cv2.putText(vis, f"dR: [{self.last_dR[0]:.4f}, {self.last_dR[1]:.4f}, {self.last_dR[2]:.4f}]deg", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dc, 1); y += 24
        if self.offset_injected and self.T_at_injection is not None:
            fs = self.frame_count - self.offset_injection_frame
            t_err = (self.T[:3,3] - self.T_at_injection[:3,3]) * 100
            r_err = Rotation.from_matrix(self.T[:3,:3] @ self.T_at_injection[:3,:3].T).as_euler('xyz', degrees=True)
            it = np.linalg.norm(self.config.trans_offset)*100; ir = np.linalg.norm(self.config.rot_offset)
            tr = 100*(1-np.linalg.norm(t_err)/max(it,0.01)) if it > 0.01 else 100
            rr = 100*(1-np.linalg.norm(r_err)/max(ir,0.01)) if ir > 0.01 else 100
            cv2.putText(vis, f"OFFSET TEST (+{fs})", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2); y += 18
            cv2.putText(vis, f"T err: [{t_err[0]:.2f}, {t_err[1]:.2f}, {t_err[2]:.2f}]cm", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2); y += 16
            cv2.putText(vis, f"R err: [{r_err[0]:.2f}, {r_err[1]:.2f}, {r_err[2]:.2f}]deg", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2); y += 16
            tc = (0,255,0) if tr > 50 else (0,165,255) if tr > 0 else (0,0,255)
            cv2.putText(vis, f"T rec: {tr:.1f}% | R rec: {rr:.1f}%", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 2)
        else:
            cv2.putText(vis, "'o'-offset 'r'-reset 'p'-plots", (w-390,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
        if result['updated']: cv2.putText(vis, "UPDATED", (w-100,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return vis
    
    def save(self, path="calibration_online.json"):
        with open(path,'w') as f: json.dump({'T':self.T.tolist(), 'updates':self.update_count}, f, indent=2)
        print(f"  Calibration saved: {path}")
    
    def save_plots(self, prefix="calibration"):
        self.metrics.generate_plots(f"{prefix}_metrics.png")
        self.metrics.save_data(f"{prefix}_metrics.json")

class StreamingLoader:
    def __init__(self, path): self.files = sorted(Path(path).glob("*.mcap")); print(f"Found {len(self.files)} MCAP files")
    def stream(self, lidar_topic="/ouster/points", cam_topic="/cam_sync/cam0/image_raw/compressed", sync_thresh=0.05):
        decoder = DecoderFactory(); lidar_buf, cam_buf = deque(maxlen=5), deque(maxlen=10)
        for f in self.files:
            print(f"\nStreaming: {f.name}")
            try:
                with open(f,'rb') as fp:
                    for schema, channel, msg in make_reader(fp, decoder_factories=[decoder]).iter_messages(topics=[lidar_topic, cam_topic]):
                        try:
                            ros_msg = decoder.decoder_for(channel.message_encoding, schema)(msg.data)
                            ts = msg.log_time / 1e9
                            if channel.topic == lidar_topic:
                                pts = parse_pc2(ros_msg)
                                if pts is not None and len(pts) > 100:
                                    lidar_buf.append((ts, pts))
                                    if cam_buf:
                                        best = min(cam_buf, key=lambda x: abs(x[0]-ts))
                                        if abs(best[0]-ts) < sync_thresh: yield {'lidar':pts, 'camera':best[1]}
                            elif channel.topic == cam_topic:
                                img = parse_img(ros_msg)
                                if img is not None: cam_buf.append((ts, img))
                        except: continue
            except Exception as e: print(f"  Error: {e}")

def parse_pc2(msg):
    fields = {f.name: f.offset for f in msg.fields}
    data = msg.data.tobytes() if hasattr(msg.data, 'tobytes') else bytes(msg.data)
    n = msg.width * msg.height
    try:
        names, fmts, offs = ['x','y','z'], ['<f4','<f4','<f4'], [fields.get('x',0), fields.get('y',4), fields.get('z',8)]
        if 'intensity' in fields or 'signal' in fields:
            names.append('intensity'); fmts.append('<f4'); offs.append(fields.get('intensity', fields.get('signal')))
        dt = np.dtype({'names':names, 'formats':fmts, 'offsets':offs, 'itemsize':msg.point_step})
        s = np.frombuffer(data, dtype=dt, count=n)
        cols = [s['x'], s['y'], s['z']]
        if 'intensity' in names: cols.append(s['intensity'])
        pts = np.column_stack(cols)
        valid = (np.linalg.norm(pts[:,:3], axis=1) > 0.5) & ~np.isnan(pts[:,:3]).any(axis=1)
        return pts[valid].astype(np.float32)
    except: return None

def parse_img(msg):
    data = bytes(msg.data) if hasattr(msg.data, '__iter__') else msg.data
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def parse_offset(s):
    try:
        vals = [float(x.strip()) for x in s.split(',')]
        return np.array(vals if len(vals)==3 else [vals[0],0,0])
    except: return np.array([0.0,0.0,0.0])

def main():
    import sys
    DATA_PATH = "."; trans_offset = np.array([0.02,0.0,0.0]); rot_offset = np.array([0.0,0.0,0.0])
    args = sys.argv[1:]; i = 0
    while i < len(args):
        if args[i] == '--trans-offset' and i+1 < len(args):
            trans_offset = parse_offset(args[i+1])
            if np.any(np.abs(trans_offset) > 0.5): trans_offset /= 100.0
            i += 2
        elif args[i] == '--rot-offset' and i+1 < len(args): rot_offset = parse_offset(args[i+1]); i += 2
        elif not args[i].startswith('-'): DATA_PATH = args[i]; i += 1
        else: i += 1
    
    print("="*60 + "\nOnline Camera-LiDAR Calibration with Metrics\n" + "="*60)
    print(f"  Trans offset: {trans_offset*100}cm | Rot offset: {rot_offset}deg\n" + "="*60)
    
    config = OnlineCalibConfig(); config.trans_offset = trans_offset; config.rot_offset = rot_offset
    calibrator = OnlineCalibrator(config); loader = StreamingLoader(DATA_PATH)
    h, w = config.camera.height, config.camera.width
    out = cv2.VideoWriter('calibration_online.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (w,h))
    
    for i, frame in enumerate(loader.stream()):
        result = calibrator.process(frame['lidar'], frame['camera'])
        vis = result['vis']
        if vis.shape[:2] != (h,w): vis = cv2.resize(vis, (w,h))
        out.write(vis)
        try:
            cv2.imshow("Online Calibration", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('o'): calibrator.inject_offset()
            elif key == ord('r'): calibrator.reset_calibration()
            elif key == ord('p'): calibrator.save_plots()
        except: pass
        if (i+1) % 10 == 0: print(f"  Frame {i+1} | Cost: {result['edge_cost']:.1f}px | Updates: {calibrator.update_count}")
    
    out.release()
    try: cv2.destroyAllWindows()
    except: pass
    calibrator.save(); calibrator.save_plots()
    print(f"\nFinal T:\n{calibrator.T}")
    print("\nOutputs: calibration_online.json, calibration_online.mp4, calibration_metrics.png, calibration_metrics.json")

if __name__ == "__main__": main()