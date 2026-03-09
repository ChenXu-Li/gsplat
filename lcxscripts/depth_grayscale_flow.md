# depth_mode: grayscale 时的深度渲染和可视化流程

## 完整流程概览

当 `depth_mode: grayscale` 时，深度渲染和可视化的完整流程如下：

```
1. 深度渲染 (render_cubemap_for_view)
   ↓
2. Cubemap 到全景图转换 (cubemap_to_equirect)
   ↓
3. 深度可视化 (visualize_depth_map_gsplat_style)
   ↓
4. 输出结果
```

---

## 详细流程

### 阶段 1: 深度渲染 (render_cubemap_for_view)

**函数**: `render_cubemap_for_view()`

**输入**:
- `splats`: 3D Gaussian Splatting 参数（means, scales, quats, opacities, colors）
- `c2w_center`: 中心相机位姿（camera-to-world 矩阵）
- `face_size`: Cubemap 每面的分辨率（如 1024）
- `near_plane`, `far_plane`: 近远平面
- `depth_type`: "expected" (ED) 或 "accumulated" (D)

**处理过程**:

1. **生成 6 个 Cubemap 相机位姿**
   - 根据中心位姿生成 6 个面的相机位姿（posx, negx, posy, negy, posz, negz）
   - 每个面使用 90° FOV 的针孔相机模型

2. **调用 gsplat rasterization 渲染**
   ```python
   render_mode = "RGB+ED"  # 如果 depth_type="expected"
   # 或
   render_mode = "RGB+D"   # 如果 depth_type="accumulated"
   ```
   - 渲染 RGB 和深度通道
   - 深度值以**世界空间单位**存储（不是归一化的 [0,1]）
   - RGB 通道被 clamp 到 [0, 1]，但**深度值不被 clamp**

3. **提取深度图**
   ```python
   depth = colors[..., 3:4]  # [6, H, W, 1] - 世界空间深度值
   depth_np = depth.squeeze(-1)  # [6, H, W] - 转换为 numpy
   ```

**输出**:
- `cube_faces`: Dict[str, np.ndarray] - 6 个面的 RGB 图像 [H, W, 3] uint8
- `cube_depths`: Dict[str, np.ndarray] - 6 个面的深度图 [H, W] float32（世界空间单位）

**关键点**:
- ✅ 深度值保持原始世界空间单位（可能很大，如 0.01 ~ 1e10）
- ✅ 深度值不被归一化或 clamp
- ✅ 无效像素（无几何体）深度值为 0 或 ≤ 0

---

### 阶段 2: Cubemap 到全景图转换 (cubemap_to_equirect)

**函数**: `cubemap_to_equirect()`

**输入**:
- `cube_faces`: 6 个面的 RGB 图像
- `cube_depths`: 6 个面的深度图
- `pano_h`, `pano_w`: 全景图分辨率（如 1024x2048）

**处理过程**:

1. **等距圆柱投影 (Equirectangular Projection)**
   - 将 6 个 cubemap 面映射到等距圆柱投影
   - 使用最近邻采样（nearest neighbor sampling）

2. **深度值传递**
   - 深度值**直接传递**，不进行任何变换
   - 保持世界空间单位

**输出**:
- `pano`: [H, W, 3] uint8 RGB 全景图
- `pano_depth`: [H, W] float32 深度全景图（世界空间单位）

**关键点**:
- ✅ 深度值仍然保持世界空间单位
- ✅ 无效像素深度值仍为 0 或 ≤ 0

---

### 阶段 3: 深度可视化 (visualize_depth_map_gsplat_style)

**函数**: `visualize_depth_map_gsplat_style()`

**输入**:
- `depth`: [H, W] float32 深度图（世界空间单位）
- `color_space`: "linear" 或 "log"（由配置 `depth_color_space` 决定）

**处理过程**:

#### 步骤 1: 创建有效掩码
```python
valid_mask = depth > 0  # 只处理有效深度值（> 0）
```

#### 步骤 2: 颜色空间变换（可选）
根据 `depth_color_space` 配置：

**如果 `depth_color_space: "linear"`**:
```python
depth_vis = depth.copy()  # 不进行变换
```

**如果 `depth_color_space: "log"`**:
```python
depth_vis[valid_mask] = np.log(depth_vis[valid_mask])  # 对数变换
```
- 目的：在对数空间下可视化，更适合观察大范围深度变化
- 效果：压缩大深度值，扩展小深度值

#### 步骤 3: 归一化到 [0, 1]
```python
depth_min = depth_vis[valid_mask].min()
depth_max = depth_vis[valid_mask].max()
depth_norm = (depth_vis - depth_min) / (depth_max - depth_min)  # [0, 1]
depth_norm = np.clip(depth_norm, 0.0, 1.0)
```
- 使用 min/max 归一化（与 gsplat render_traj 一致）
- 只对有效像素进行归一化

#### 步骤 4: 转换为灰度图
```python
depth_gray = (depth_norm * 255.0).astype(np.uint8)  # [H, W] uint8 [0, 255]
depth_gray = np.stack([depth_gray, depth_gray, depth_gray], axis=-1)  # [H, W, 3]
```
- 将归一化值 [0, 1] 缩放到 [0, 255]
- 重复 3 次生成 RGB 灰度图（3 个通道值相同）

#### 步骤 5: 处理无效像素
```python
depth_gray[~valid_mask] = 0  # 无效像素设为黑色
```

**输出**:
- `depth_gray`: [H, W, 3] uint8 灰度深度可视化图

**关键点**:
- ✅ 灰度值范围：[0, 255]，其中 0 = 最近，255 = 最远
- ✅ 无效像素（无几何体）显示为黑色 (0, 0, 0)
- ✅ 与 gsplat 的 render_traj 可视化风格一致

---

### 阶段 4: 输出结果

**输出文件**:

1. **Cubemap Grid** (`{view_tag}_cubemap_grid.png`)
   - 2 行 × 6 列网格
   - 第 1 行：6 个面的 RGB 图像
   - 第 2 行：6 个面的深度可视化（灰度图）

2. **Combined Panorama** (`{view_tag}_pano_combined.png`)
   - 上下拼接的全景图
   - 上半部分：RGB 全景图
   - 下半部分：深度可视化（灰度图）

---

## 配置项影响

### `depth_mode: grayscale`
- ✅ 使用 `visualize_depth_map_gsplat_style()` 函数
- ✅ 输出灰度图（3 通道，值相同）
- ❌ **不使用** `use_disparity` 配置（只在 colored 模式生效）

### `depth_color_space: linear` 或 `log`
- ✅ **在 grayscale 模式下生效**
- `linear`: 直接使用原始深度值进行归一化
- `log`: 先应用对数变换，再归一化（更适合大范围深度变化）

### `depth_type: expected` 或 `accumulated`
- ✅ 影响渲染阶段的深度计算方式
- `expected`: 使用期望深度 (ED)，加权平均，更稳定
- `accumulated`: 使用实际深度 (D)，直接累加，可能更精确但边缘不稳定

---

## 数据流示例

假设深度值范围：`[0.01, 100.0]`（世界空间单位）

### 情况 1: `depth_color_space: linear`

```
原始深度: [0.01, 100.0]
    ↓
颜色空间变换: 无变换
    ↓
depth_vis: [0.01, 100.0]
    ↓
归一化: (depth_vis - 0.01) / (100.0 - 0.01)
    ↓
depth_norm: [0.0, 1.0]
    ↓
缩放: depth_norm * 255
    ↓
灰度图: [0, 255] (0=最近, 255=最远)
```

### 情况 2: `depth_color_space: log`

```
原始深度: [0.01, 100.0]
    ↓
颜色空间变换: log(depth)
    ↓
depth_vis: [log(0.01), log(100.0)] ≈ [-4.605, 4.605]
    ↓
归一化: (depth_vis - (-4.605)) / (4.605 - (-4.605))
    ↓
depth_norm: [0.0, 1.0]
    ↓
缩放: depth_norm * 255
    ↓
灰度图: [0, 255] (0=最近, 255=最远)
```

**注意**: 对数变换会压缩大深度值，扩展小深度值，使得近距离细节更明显。

---

## 总结

当 `depth_mode: grayscale` 时：

1. **渲染阶段**: 使用 gsplat rasterization 渲染世界空间深度值
2. **转换阶段**: Cubemap → 全景图，深度值保持不变
3. **可视化阶段**: 
   - 根据 `depth_color_space` 决定是否应用对数变换
   - Min/max 归一化到 [0, 1]
   - 转换为灰度图 [0, 255]
4. **输出**: 灰度深度可视化图（与 gsplat render_traj 风格一致）

**关键配置**:
- ✅ `depth_color_space`: 控制是否使用对数空间（在 grayscale 模式下生效）
- ❌ `use_disparity`: **不生效**（只在 colored 模式下生效）
