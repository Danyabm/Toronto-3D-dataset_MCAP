import open3d as o3d
import numpy as np
from mcap.writer import Writer
import json
import base64
from dataclasses import dataclass
import cv2
import time
import matplotlib.pyplot as plt

print("All libraries imported successfully!")
# LOAD DATA
# ============================================
file_path = '/Users/danya/Downloads/Toronto_3D/L001.ply'
pcd = o3d.io.read_point_cloud(file_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

if len(points) > 1000000:
    indices = np.random.choice(len(points), 1000000, replace=False)
    points, colors = points[indices], colors[indices]

original_center = points.mean(axis=0)
points = points - original_center
extent = points.max(axis=0) - points.min(axis=0)
z_min, z_max = points[:, 2].min(), points[:, 2].max()
z_center = (z_min + z_max) / 2

print(f"✅ Loaded {len(points):,} points")
print(f"   Z range: [{z_min:.1f}, {z_max:.1f}]m\n")
# CAMERA
# ============================================
@dataclass
class PinholeCamera:
    width: int = 1280
    height: int = 720
    fx: float = 400.0
    fy: float = 400.0
    cx: float = 640.0
    cy: float = 360.0
    
    def render_image(self, points_3d, colors, camera_pose):
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        points_cam = (points_3d - t) @ R.T
        
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        points_cam = points_cam[valid]
        valid_colors = colors[valid]
        
        z = points_cam[:, 2]
        u = (self.fx * points_cam[:, 0] / z + self.cx).astype(np.int32)
        v = (self.fy * points_cam[:, 1] / z + self.cy).astype(np.int32)
        
        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        if not np.any(in_bounds):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]
        valid_colors = valid_colors[in_bounds]
        
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        sort_idx = np.argsort(-z)
        img[v[sort_idx], u[sort_idx]] = (valid_colors[sort_idx] * 255).astype(np.uint8)
        
        return img

camera = PinholeCamera()
print(f"✅ Camera: {camera.width}x{camera.height}")
# LIDAR
# ============================================
class SimpleLiDAR:
    def __init__(self):
        self.num_lasers = 64
        self.horizontal_res = 0.18
        self.max_range = 60.0  # Increased range
        self.vertical_fov = (-25, 15)
    
    def scan(self, points_3d, lidar_pose):
        R = lidar_pose[:3, :3]
        t = lidar_pose[:3, 3]
        points_lidar = (points_3d - t) @ R.T
        
        x, y, z = points_lidar[:, 0], points_lidar[:, 1], points_lidar[:, 2]
        range_dist = np.sqrt(x**2 + y**2 + z**2)
        
        valid = (range_dist > 0.1) & (range_dist < self.max_range)
        if not np.any(valid):
            return np.zeros((0, 3))
        
        x, y, z = x[valid], y[valid], z[valid]
        range_dist = range_dist[valid]
        points_world = points_3d[valid]
        
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(np.clip(z / range_dist, -1, 1))
        
        min_elev = np.deg2rad(self.vertical_fov[0])
        max_elev = np.deg2rad(self.vertical_fov[1])
        elev_valid = (elevation >= min_elev) & (elevation <= max_elev)
        if not np.any(elev_valid):
            return np.zeros((0, 3))
        
        azimuth, elevation, range_dist = azimuth[elev_valid], elevation[elev_valid], range_dist[elev_valid]
        points_world = points_world[elev_valid]
        
        num_horizontal = int(360 / self.horizontal_res)
        az_bins = ((azimuth + np.pi) / (2*np.pi) * num_horizontal).astype(np.int32)
        el_bins = ((elevation - min_elev) / (max_elev - min_elev) * self.num_lasers).astype(np.int32)
        el_bins = np.clip(el_bins, 0, self.num_lasers - 1)
        bin_idx = el_bins * num_horizontal + az_bins
        
        unique_bins = np.unique(bin_idx)
        scan_points = [points_world[np.where(bin_idx == b)[0][np.argmin(range_dist[bin_idx == b])]] for b in unique_bins]
        
        return np.array(scan_points) if scan_points else np.zeros((0, 3))

lidar = SimpleLiDAR()
print(f"✅ LiDAR: {lidar.max_range}m range\n")
# TRAJECTORY - ELEVATED AND TILTED DOWN
# ============================================
def create_aerial_trajectory(center, radius, height_above, tilt_down_deg, num_frames, duration=15.0):
    """Create trajectory with camera ABOVE scene, tilted DOWN"""
    timestamps = np.linspace(0, duration, num_frames)
    poses = []
    
    tilt_rad = np.deg2rad(tilt_down_deg)
    
    for t in timestamps:
        angle = (t / duration) * 2 * np.pi
        
        # Position ABOVE the scene
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = height_above  # Elevated above scene
        position = np.array([x, y, z])
        
        # Look at center point BELOW camera
        look_at = np.array([center[0], center[1], center[2]])
        
        # Forward vector (toward center)
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        # Build camera frame
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([right, up, forward])
        pose[:3, 3] = position
        poses.append(pose)
    
    return timestamps, poses

# ⭐ KEY FIX: Position camera ABOVE scene looking DOWN
radius = 50.0  # 50m from center
height_above = z_max + 15.0  # 15m above highest point
tilt_down = 45  # Look down at 45 degrees
num_frames = 150

print(f"🎬 Trajectory:")
print(f"   Radius: {radius:.0f}m")
print(f"   Height: {height_above:.1f}m (above scene)")
print(f"   Camera looks DOWN at scene")
print(f"   Frames: {num_frames}\n")

timestamps, poses = create_aerial_trajectory(
    center=np.array([0, 0, z_center]),
    radius=radius,
    height_above=height_above,
    tilt_down_deg=tilt_down,
    num_frames=num_frames,
    duration=15.0
)

# Test visibility
test_pose = poses[75]
test_R, test_t = test_pose[:3, :3], test_pose[:3, 3]
test_pts_cam = (points - test_t) @ test_R.T
test_valid = test_pts_cam[:, 2] > 0.1

print(f"🔍 Visibility test:")
print(f"   Camera position: [{test_t[0]:.0f}, {test_t[1]:.0f}, {test_t[2]:.0f}]")
print(f"   Points in front: {np.sum(test_valid):,} ({np.sum(test_valid)/len(points)*100:.0f}%)\n")
# RENDER
# ============================================
print("📸 Rendering...")
camera_images, lidar_scans = [], []

start = time.time()
for i, pose in enumerate(poses):
    if i % 30 == 0:
        print(f"  {i}/{num_frames}...")
    camera_images.append(camera.render_image(points, colors, pose))
    lidar_scans.append(lidar.scan(points, pose))

print(f"✅ Rendered in {time.time()-start:.0f}s")

# Check sample
sample_nonzero = np.count_nonzero(camera_images[75])
print(f"   Sample frame: {sample_nonzero:,} pixels ({sample_nonzero/(1280*720)*100:.1f}%)\n")
# VERIFY
# ============================================
sample_idx = 75
non_zero = np.count_nonzero(camera_images[sample_idx])
total_pixels = camera_images[sample_idx].size
lidar_pts = len(lidar_scans[sample_idx])

print(f"📊 Frame {sample_idx}:")
print(f"   Camera: {non_zero:,} / {total_pixels:,} pixels ({non_zero/total_pixels*100:.1f}%)")
print(f"   LiDAR: {lidar_pts:,} points")

if non_zero > 200000:
    print("   ✅ EXCELLENT!")
elif non_zero > 50000:
    print("   ✅ GOOD!")
elif non_zero > 10000:
    print("   ⚠️ Needs improvement")
else:
    print("   ❌ Still issues")

# Show
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.imshow(camera_images[sample_idx])
ax1.set_title(f'Camera ({non_zero:,} pixels, {non_zero/total_pixels*100:.1f}%)')
ax1.axis('off')

scan = lidar_scans[sample_idx]
if len(scan) > 0:
    sc = ax2.scatter(scan[:, 0], scan[:, 1], c=scan[:, 2], s=2, cmap='viridis')
    ax2.set_title(f'LiDAR ({len(scan):,} points)')
    ax2.axis('equal')
    plt.colorbar(sc, ax=ax2, label='Z (m)')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("🔬 COMPREHENSIVE CAMERA DEBUG\n" + "="*70)

sample_idx = 75
pose = poses[sample_idx]
R = pose[:3, :3]
t = pose[:3, 3]

print(f"Camera Pose Analysis:")
print(f"  Position: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
print(f"  Distance from origin: {np.linalg.norm(t):.2f}m")
print(f"  Forward vector: {pose[:3, 2]}")
print(f"  Scene Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
print(f"  Camera Z: {t[2]:.2f}")

# Transform points to camera frame
points_cam = (points - t) @ R.T

print(f"\n" + "="*70)
print(f"Camera Frame Analysis (all {len(points):,} points):")
print(f"  X: [{points_cam[:, 0].min():.2f}, {points_cam[:, 0].max():.2f}]")
print(f"  Y: [{points_cam[:, 1].min():.2f}, {points_cam[:, 1].max():.2f}]")
print(f"  Z: [{points_cam[:, 2].min():.2f}, {points_cam[:, 2].max():.2f}]")

# Check front/back split
valid = points_cam[:, 2] > 0.1
print(f"\n  In front (Z>0.1): {np.sum(valid):,} ({np.sum(valid)/len(points)*100:.1f}%)")
print(f"  Behind (Z<=0.1): {np.sum(~valid):,} ({np.sum(~valid)/len(points)*100:.1f}%)")

if np.sum(valid) > 0:
    pts_front = points_cam[valid]
    colors_front = colors[valid]
    
    print(f"\n" + "="*70)
    print(f"Points In Front Analysis ({np.sum(valid):,} points):")
    print(f"  Z range: [{pts_front[:, 2].min():.2f}, {pts_front[:, 2].max():.2f}]")
    print(f"  Z mean: {pts_front[:, 2].mean():.2f}")
    
    # Project
    z = pts_front[:, 2]
    u = camera.fx * pts_front[:, 0] / z + camera.cx
    v = camera.fy * pts_front[:, 1] / z + camera.cy
    
    print(f"\n  Projection before clipping:")
    print(f"    U: [{u.min():.1f}, {u.max():.1f}] (image: 0-{camera.width})")
    print(f"    V: [{v.min():.1f}, {v.max():.1f}] (image: 0-{camera.height})")
    
    # Check bounds
    in_bounds = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    print(f"\n  In image bounds: {np.sum(in_bounds):,} ({np.sum(in_bounds)/len(u)*100:.2f}%)")
    
    if np.sum(in_bounds) > 0:
        print(f"\n✅ {np.sum(in_bounds):,} points SHOULD be visible")
        print(f"❌ But only 525 pixels are rendered")
        print(f"\n🔍 This suggests a RENDERING BUG, not a geometry problem!")
        
        # Check if colors are valid
        colors_in_bounds = colors_front[in_bounds]
        print(f"\n  Color check:")
        print(f"    Min: {colors_in_bounds.min():.3f}")
        print(f"    Max: {colors_in_bounds.max():.3f}")
        print(f"    Mean: {colors_in_bounds.mean():.3f}")
        print(f"    All zeros?: {np.all(colors_in_bounds == 0)}")
        
        # Sample render test
        print(f"\n  Testing manual render of first 10k points...")
        u_test = u[in_bounds][:10000].astype(np.int32)
        v_test = v[in_bounds][:10000].astype(np.int32)
        c_test = colors_in_bounds[:10000]
        
        test_img = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        test_img[v_test, u_test] = (c_test * 255).astype(np.uint8)
        
        test_nonzero = np.count_nonzero(test_img)
        print(f"    Manual render: {test_nonzero:,} non-zero pixels")
        print(f"    Actual render: 525 non-zero pixels")
        
        if test_nonzero > 5000:
            print(f"\n🚨 PROBLEM FOUND: Manual render works, but camera.render_image() doesn't!")
            print(f"   The issue is in the render_image() function's depth sorting loop")
    else:
        print(f"\n❌ PROBLEM: No points project into image!")
        print(f"   Camera FOV is too narrow or camera is too far")
        
        # Calculate what FOV would work
        angles_h = np.arctan2(pts_front[:, 0], pts_front[:, 2])
        angles_v = np.arctan2(pts_front[:, 1], pts_front[:, 2])
        
        needed_fov_h = np.rad2deg(angles_h.max() - angles_h.min())
        needed_fov_v = np.rad2deg(angles_v.max() - angles_v.min())
        
        current_fov_h = np.rad2deg(2 * np.arctan(camera.width / (2 * camera.fx)))
        current_fov_v = np.rad2deg(2 * np.arctan(camera.height / (2 * camera.fy)))
        
        print(f"\n  FOV Analysis:")
        print(f"    Current FOV: {current_fov_h:.1f}° × {current_fov_v:.1f}°")
        print(f"    Needed FOV: {needed_fov_h:.1f}° × {needed_fov_v:.1f}°")
        print(f"    Recommendation: Use wider FOV (lower fx/fy values)")
else:
    print(f"\n❌ CRITICAL: No points in front of camera at all!")
    print(f"   Camera orientation is completely wrong")

# Visualize in 3D
fig = plt.figure(figsize=(15, 10))

# 3D scene view
ax1 = fig.add_subplot(221, projection='3d')
sample_pts = points[::200]
ax1.scatter(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], 
           c='lightgray', s=0.5, alpha=0.5)
ax1.scatter(*t, c='red', s=200, marker='o', label='Camera', edgecolors='black', linewidths=2)
forward = pose[:3, 2] * 30
ax1.plot([t[0], t[0]+forward[0]], [t[1], t[1]+forward[1]], [t[2], t[2]+forward[2]], 
        'b-', linewidth=3, label='Forward')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title('3D Scene View')
ax1.legend()

# Top view
ax2 = fig.add_subplot(222)
ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=0.1, alpha=0.3)
ax2.scatter(t[0], t[1], c='red', s=200, marker='o', edgecolors='black', linewidths=2)
ax2.arrow(t[0], t[1], forward[0], forward[1], 
         head_width=8, head_length=5, fc='blue', ec='blue', linewidth=2)
ax2.set_xlabel('X'); ax2.set_ylabel('Y')
ax2.set_title('Top-Down View')
ax2.axis('equal')
ax2.grid(True, alpha=0.3)

# Side view (X-Z)
ax3 = fig.add_subplot(223)
ax3.scatter(points[:, 0], points[:, 2], c='lightgray', s=0.1, alpha=0.3)
ax3.scatter(t[0], t[2], c='red', s=200, marker='o', edgecolors='black', linewidths=2)
ax3.axhline(t[2], color='red', linestyle='--', alpha=0.5, label='Camera height')
ax3.set_xlabel('X'); ax3.set_ylabel('Z')
ax3.set_title('Side View (X-Z)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Current rendered image
ax4 = fig.add_subplot(224)
ax4.imshow(camera_images[sample_idx])
ax4.set_title(f'Rendered Image ({np.count_nonzero(camera_images[sample_idx]):,} pixels)')
ax4.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY:")
if np.sum(valid) > 100000 and np.sum(in_bounds) > 10000:
    print("✅ Geometry is CORRECT - points are in front and project into frame")
    print("❌ Rendering function has a BUG - depth sorting or pixel assignment issue")
    print("\n💡 SOLUTION: Fix the render_image() depth buffer implementation")
elif np.sum(in_bounds) < 1000:
    print("❌ FOV is too narrow - camera can't see enough of the scene")
    print("\n💡 SOLUTION: Reduce fx and fy to widen field of view")
else:
    print("❌ Camera positioning or orientation issue")
    print("\n💡 SOLUTION: Check trajectory height and forward direction")
    # MCAP EXPORT
# ============================================
def export_foxglove_mcap(path, timestamps, images, scans):
    print(f"📦 Exporting to {path}...\n")
    
    import datetime
    base_time = datetime.datetime(2024, 11, 21, 12, 0, 0).timestamp()
    
    with open(path, 'wb') as f:
        writer = Writer(f)
        writer.start()
        
        img_schema = json.dumps({
            "type": "object",
            "properties": {
                "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
                "frame_id": {"type": "string"},
                "format": {"type": "string"},
                "data": {"type": "string", "contentEncoding": "base64"}
            }
        })
        
        pc_schema = json.dumps({
            "type": "object",
            "properties": {
                "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
                "frame_id": {"type": "string"},
                "pose": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
                        "orientation": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}}}
                    }
                },
                "points": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}}}
            }
        })
        
        img_sid = writer.register_schema(name="foxglove.CompressedImage", encoding="jsonschema", data=img_schema.encode())
        pc_sid = writer.register_schema(name="foxglove.PointCloud", encoding="jsonschema", data=pc_schema.encode())
        
        img_ch = writer.register_channel(topic="/camera/image_compressed", message_encoding="json", schema_id=img_sid)
        pc_ch = writer.register_channel(topic="/lidar/points", message_encoding="json", schema_id=pc_sid)
        
        for i, (ts, img, scan) in enumerate(zip(timestamps, images, scans)):
            if i % 30 == 0 or i == len(timestamps) - 1:
                print(f"  {(i+1)/len(timestamps)*100:.0f}%")
            
            timestamp = base_time + ts
            sec, nsec = int(timestamp), int((timestamp - int(timestamp)) * 1e9)
            time_ns = int(timestamp * 1e9)
            
            success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                img_msg = json.dumps({
                    "timestamp": {"sec": sec, "nsec": nsec},
                    "frame_id": "camera",
                    "format": "jpeg",
                    "data": base64.b64encode(buffer.tobytes()).decode('ascii')
                })
                writer.add_message(img_ch, time_ns, img_msg.encode('utf-8'), time_ns)
            
            if len(scan) == 0:
                scan = np.zeros((1, 3))
            
            pc_msg = json.dumps({
                "timestamp": {"sec": sec, "nsec": nsec},
                "frame_id": "world",
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                },
                "points": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in scan.astype(np.float64)]
            })
            writer.add_message(pc_ch, time_ns, pc_msg.encode('utf-8'), time_ns)
        
        writer.finish()
    
    import os
    print(f"\n✅ Done: {os.path.getsize(path)/(1024*1024):.1f} MB")
    print("\n🦊 Foxglove: Camera should show aerial view of Toronto streets!")

export_foxglove_mcap('toronto_3d_simulation.mcap', timestamps, camera_images, lidar_scans)
print("\n🎉 COMPLETE!")