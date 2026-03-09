# depth_mode: colored 时的深度渲染和可视化流程

## 完整流程概览

当 `depth_mode: colored` 时，深度渲染和可视化的完整流程如下：

```
1. 深度渲染 (render_cubemap_for_view)
   ↓
2. Cubemap 到全景图转换 (cubemap_to_equirect)
   ↓
3. 深度可视化 (visualize_depth_map_colored)
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

### 阶段 3: 深度可视化 (visualize_depth_map_colored)

**函数**: `visualize_depth_map_colored()`

**输入**:
- `depth`: [H, W] float32 深度图（世界空间单位）
- `use_disparity`: bool - 是否转换为视差 (1/depth)
- `color_space`: "linear" 或 "log"（由配置 `depth_color_space` 决定）
- `cmap`: Matplotlib colormap 名称（如 "Spectral", "turbo", "viridis"）
- `percentile`: float - 用于计算 min/max 的百分位数（默认 2.0）

**处理过程**:

#### 步骤 1: 创建有效掩码
```python
valid_mask = depth > 0  # 只处理有效深度值（> 0）
```

#### 步骤 2: 视差变换（可选）
根据 `use_disparity` 配置：

**如果 `use_disparity: true`** (默认，DA3 风格):
```python
depth[valid_mask] = 1.0 / depth[valid_mask]  # 转换为视差
```
- **目的**: 处理不均匀的深度分布，让近处物体有更高的值
- **效果**: 
  - 近处物体（小深度值）→ 大视差值
  - 远处物体（大深度值）→ 小视差值
  - 使得近距离细节更容易观察

**如果 `use_disparity: false`**:
```python
# 不进行变换，保持原始深度值
```

#### 步骤 3: 颜色空间变换（可选）
根据 `depth_color_space` 配置：

**如果 `depth_color_space: "linear"`**:
```python
# 不进行变换
```

**如果 `depth_color_space: "log"`**:
```python
depth[valid_mask] = np.log(depth[valid_mask])  # 对数变换
```
- **目的**: 在对数空间下可视化，更适合观察大范围深度变化
- **效果**: 压缩大深度值，扩展小深度值

**注意**: 如果同时使用 `use_disparity` 和 `log`，执行顺序是：
1. 先执行视差变换：`depth = 1.0 / depth`
2. 再执行对数变换：`depth = log(depth)`

#### 步骤 4: 百分位数归一化到 [0, 1]
```python
# 使用百分位数计算 min/max（避免极端值影响）
depth_min = np.percentile(valid_depth, percentile)  # 默认 2%
depth_max = np.percentile(valid_depth, 100.0 - percentile)  # 默认 98%

# 归一化
depth_norm = (depth - depth_min) / (depth_max - depth_min)  # [0, 1]
depth_norm = np.clip(depth_norm, 0.0, 1.0)
```
- **与 grayscale 模式的区别**: 
  - Grayscale: 使用 min/max 归一化
  - Colored: 使用百分位数归一化（更鲁棒，避免极端值影响）
- **目的**: 排除 2% 的最小值和 2% 的最大值，使可视化更稳定

#### 步骤 5: 颜色反转
```python
depth_norm = 1.0 - depth_norm  # 反转
```
- **目的**: 使得近处物体显示为红色，远处物体显示为蓝色（Spectral colormap）
- **逻辑**: 
  - 当 `use_disparity=True` 时，近处物体有更高的视差值
  - 反转后，高值（近处）→ 低归一化值 → 红色
  - 低值（远处）→ 高归一化值 → 蓝色

#### 步骤 6: 应用 Colormap
```python
cm = matplotlib.colormaps[cmap]  # 例如 "Spectral", "turbo", "viridis"
img_colored = cm(depth_norm[None], bytes=False)[:, :, :, 0:3]  # [1, H, W, 3]
img_colored = (img_colored[0] * 255.0).astype(np.uint8)  # [H, W, 3] uint8
```
- 将归一化值 [0, 1] 映射到 colormap 颜色
- 默认使用 "Spectral" colormap（红-黄-绿-青-蓝）

#### 步骤 7: 处理无效像素
```python
img_colored[~valid_mask] = 0  # 无效像素设为黑色
```

**输出**:
- `img_colored`: [H, W, 3] uint8 彩色深度可视化图

**关键点**:
- ✅ 使用百分位数归一化（更鲁棒）
- ✅ 支持视差变换（`use_disparity`）
- ✅ 支持对数空间变换（`depth_color_space`）
- ✅ 颜色反转：近处红色，远处蓝色
- ✅ 与 depth-anything-3 的可视化风格一致

---

### 阶段 4: 输出结果

**输出文件**:

1. **Cubemap Grid** (`{view_tag}_cubemap_grid.png`)
   - 2 行 × 6 列网格
   - 第 1 行：6 个面的 RGB 图像
   - 第 2 行：6 个面的深度可视化（彩色图）

2. **Combined Panorama** (`{view_tag}_pano_combined.png`)
   - 上下拼接的全景图
   - 上半部分：RGB 全景图
   - 下半部分：深度可视化（彩色图）

---

## 配置项影响

### `depth_mode: colored`
- ✅ 使用 `visualize_depth_map_colored()` 函数
- ✅ 输出彩色图（使用 colormap）
- ✅ **使用** `use_disparity` 配置（在 colored 模式下生效）

### `use_disparity: true` 或 `false`
- ✅ **在 colored 模式下生效**
- `true` (默认): 将深度转换为视差 (1/depth)，让近处物体有更高的值
- `false`: 直接使用原始深度值

### `depth_color_space: linear` 或 `log`
- ✅ **在 colored 模式下生效**
- `linear`: 不进行对数变换
- `log`: 应用对数变换（在视差变换之后，如果启用）

### `depth_cmap: "Spectral"` 等
- ✅ **只在 colored 模式下生效**
- 可用的 colormap: "Spectral", "turbo", "viridis", "jet", "plasma" 等
- 默认: "Spectral"（红-黄-绿-青-蓝）

### `depth_percentile: 2.0`
- ✅ **只在 colored 模式下生效**
- 用于计算归一化范围的百分位数
- 默认 2.0 表示使用 2% 和 98% 分位数（排除极端值）

### `depth_type: expected` 或 `accumulated`
- ✅ 影响渲染阶段的深度计算方式
- `expected`: 使用期望深度 (ED)，加权平均，更稳定
- `accumulated`: 使用实际深度 (D)，直接累加，可能更精确但边缘不稳定

---

## 数据流示例

假设深度值范围：`[0.01, 100.0]`（世界空间单位）

### 情况 1: `use_disparity: true`, `depth_color_space: linear`

```
原始深度: [0.01, 100.0]
    ↓
视差变换: 1.0 / depth
    ↓
视差值: [100.0, 0.01]  (近处大，远处小)
    ↓
颜色空间变换: 无变换
    ↓
depth_vis: [100.0, 0.01]
    ↓
百分位数归一化: (depth_vis - percentile_min) / (percentile_max - percentile_min)
    ↓
depth_norm: [1.0, 0.0] (反转前)
    ↓
颜色反转: 1.0 - depth_norm
    ↓
depth_norm_inv: [0.0, 1.0] (近处→0→红色, 远处→1→蓝色)
    ↓
应用 colormap (Spectral)
    ↓
彩色图: 近处红色，远处蓝色
```

### 情况 2: `use_disparity: true`, `depth_color_space: log`

```
原始深度: [0.01, 100.0]
    ↓
视差变换: 1.0 / depth
    ↓
视差值: [100.0, 0.01]
    ↓
颜色空间变换: log(depth)
    ↓
depth_vis: [log(100.0), log(0.01)] ≈ [4.605, -4.605]
    ↓
百分位数归一化
    ↓
depth_norm: [1.0, 0.0] (反转前)
    ↓
颜色反转: 1.0 - depth_norm
    ↓
depth_norm_inv: [0.0, 1.0]
    ↓
应用 colormap
    ↓
彩色图: 近处红色，远处蓝色（对数空间，细节更明显）
```

### 情况 3: `use_disparity: false`, `depth_color_space: linear`

```
原始深度: [0.01, 100.0]
    ↓
视差变换: 无变换
    ↓
depth_vis: [0.01, 100.0]  (近处小，远处大)
    ↓
颜色空间变换: 无变换
    ↓
百分位数归一化
    ↓
depth_norm: [0.0, 1.0] (反转前)
    ↓
颜色反转: 1.0 - depth_norm
    ↓
depth_norm_inv: [1.0, 0.0] (近处→1→蓝色, 远处→0→红色)
    ↓
应用 colormap
    ↓
彩色图: 近处蓝色，远处红色（与 use_disparity=true 相反）
```

**注意**: 当 `use_disparity=false` 时，颜色映射会反转（近处蓝色，远处红色），因为深度值本身是近小远大。

---

## Grayscale vs Colored 模式对比

| 特性 | Grayscale 模式 | Colored 模式 |
|------|---------------|-------------|
| **可视化函数** | `visualize_depth_map_gsplat_style()` | `visualize_depth_map_colored()` |
| **输出类型** | 灰度图（3通道，值相同） | 彩色图（使用 colormap） |
| **归一化方法** | Min/Max 归一化 | 百分位数归一化（更鲁棒） |
| **use_disparity** | ❌ 不生效 | ✅ 生效（默认 true） |
| **depth_color_space** | ✅ 生效 | ✅ 生效 |
| **颜色映射** | 线性灰度 [0, 255] | Colormap（如 Spectral） |
| **颜色含义** | 0=最近（黑），255=最远（白） | 近处红色，远处蓝色（Spectral） |
| **风格** | gsplat render_traj 风格 | depth-anything-3 风格 |

---

## 配置建议

### 推荐配置 1: DA3 默认风格
```yaml
depth_mode: colored
use_disparity: true          # 视差变换，近处细节更明显
depth_color_space: linear    # 线性空间
depth_cmap: Spectral         # 红-蓝渐变
depth_percentile: 2.0        # 排除 2% 极端值
```
- **适用场景**: 观察近距离细节，处理不均匀深度分布

### 推荐配置 2: 大范围深度可视化
```yaml
depth_mode: colored
use_disparity: true          # 视差变换
depth_color_space: log       # 对数空间，压缩大值
depth_cmap: turbo            # Turbo colormap（更平滑）
depth_percentile: 2.0
```
- **适用场景**: 深度范围很大，需要观察整体分布

### 推荐配置 3: 直接深度可视化
```yaml
depth_mode: colored
use_disparity: false         # 不使用视差，直接使用深度
depth_color_space: linear    # 线性空间
depth_cmap: viridis          # Viridis colormap
depth_percentile: 2.0
```
- **适用场景**: 需要保持深度值的线性关系

---

## 总结

当 `depth_mode: colored` 时：

1. **渲染阶段**: 使用 gsplat rasterization 渲染世界空间深度值
2. **转换阶段**: Cubemap → 全景图，深度值保持不变
3. **可视化阶段**: 
   - **可选** 视差变换 (`use_disparity`): `depth = 1.0 / depth`
   - **可选** 对数变换 (`depth_color_space`): `depth = log(depth)`
   - 百分位数归一化到 [0, 1]（排除极端值）
   - 颜色反转（近处→红色，远处→蓝色）
   - 应用 colormap（如 Spectral）
4. **输出**: 彩色深度可视化图（与 depth-anything-3 风格一致）

**关键配置**:
- ✅ `use_disparity`: 控制是否使用视差变换（**在 colored 模式下生效**）
- ✅ `depth_color_space`: 控制是否使用对数空间（在 colored 模式下生效）
- ✅ `depth_cmap`: 选择 colormap（只在 colored 模式下生效）
- ✅ `depth_percentile`: 控制归一化范围（只在 colored 模式下生效）

**与 grayscale 模式的主要区别**:
- 使用百分位数归一化（更鲁棒）
- 支持视差变换（`use_disparity`）
- 使用 colormap 而非灰度
- 颜色反转：近处红色，远处蓝色
