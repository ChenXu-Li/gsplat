# 深度可视化逻辑梳理

## 一、整体架构

深度可视化系统分为两个主要模式：
1. **灰度模式 (grayscale)**：gsplat 风格，简单线性归一化
2. **彩色模式 (colored)**：DA3 风格，使用 colormap 映射

## 二、核心函数流程

### 1. 入口函数：`visualize_depth_map()`

```python
visualize_depth_map(
    depth: np.ndarray,           # [H, W] float32 原始深度图
    mode: str = "grayscale",     # "grayscale" 或 "colored"
    cmap: str = "Spectral",      # colormap 名称（仅 colored 模式）
    percentile: float = 2.0,      # 百分位数（仅 colored 模式）
    use_disparity: bool = True,  # 是否使用视差（仅 colored 模式）
    color_space: str = "linear", # "linear" 或 "log"
)
```

**路由逻辑**：
- `mode == "grayscale"` → 调用 `visualize_depth_map_gsplat_style()`
- `mode == "colored"` → 调用 `visualize_depth_map_colored()`

---

### 2. 灰度模式：`visualize_depth_map_gsplat_style()`

**适用场景**：gsplat 风格的简单灰度可视化

**处理流程**：
```
原始深度图 [H, W] (float32)
    ↓
1. 创建有效掩码：valid_mask = depth > 0
    ↓
2. 可选：对数变换（如果 color_space == "log"）
   depth_vis[valid_mask] = log(depth[valid_mask])
    ↓
3. 线性归一化：使用全局 min/max
   depth_min = depth_vis[valid_mask].min()
   depth_max = depth_vis[valid_mask].max()
   depth_norm = (depth_vis - depth_min) / (depth_max - depth_min)
    ↓
4. 转换为灰度图：[0, 1] → [0, 255] uint8，重复 3 通道
   depth_gray = (depth_norm * 255).astype(uint8)
   depth_gray = stack([depth_gray, depth_gray, depth_gray], axis=-1)
    ↓
5. 无效像素置黑：depth_gray[~valid_mask] = 0
    ↓
输出：[H, W, 3] uint8 灰度图
```

**特点**：
- 简单直接，使用全局 min/max
- 适合快速预览
- 对极端值敏感

---

### 3. 彩色模式：`visualize_depth_map_colored()` ⭐

**适用场景**：DA3 风格的彩色深度可视化（默认推荐）

**处理流程**：
```
原始深度图 [H, W] (float32)
    ↓
1. 创建有效掩码：valid_mask = depth > 0
    ↓
2. 【关键步骤】转换为视差（默认启用，use_disparity=True）
   depth[valid_mask] = 1.0 / depth[valid_mask]
   
   作用：
   - 解决深度分布不均问题
   - 近景细节更明显（近处视差值大，远处视差值小）
   - 这是 DA3 的默认行为
    ↓
3. 可选：对数变换（如果 color_space == "log"）
   depth[valid_mask] = log(depth[valid_mask])
   
   作用：
   - 压缩大范围深度变化
   - 更适合观察大场景
    ↓
4. 【关键步骤】百分位数归一化（避免极端值）
   valid_depth = depth[valid_mask]
   
   if len(valid_depth) > 10:
       depth_min = percentile(valid_depth, percentile)      # 默认 2%
       depth_max = percentile(valid_depth, 100 - percentile) # 默认 98%
   else:
       depth_min = valid_depth.min()
       depth_max = valid_depth.max()
   
   作用：
   - 过滤极端值（如天空、背景噪点）
   - 保留主要深度信息
   - 提高可视化对比度
    ↓
5. 归一化到 [0, 1]
   depth_range = depth_max - depth_min
   depth_norm = ((depth - depth_min) / depth_range).clip(0, 1)
    ↓
6. 【关键步骤】反转映射
   depth_norm = 1.0 - depth_norm
   
   原因：
   - 当 use_disparity=True 时，近处视差值大，远处视差值小
   - 反转后：近处 → 高值（红色），远处 → 低值（蓝色）
   - 符合直觉：近=红，远=蓝
    ↓
7. 应用 colormap
   cm = matplotlib.colormaps[cmap]  # 默认 "Spectral"
   img_colored = cm(depth_norm[None])  # [1, H, W, 3]
   img_colored = (img_colored[0] * 255).astype(uint8)
    ↓
8. 无效像素置黑：img_colored[~valid_mask] = 0
    ↓
输出：[H, W, 3] uint8 彩色深度图
```

**关键参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_disparity` | `True` | 使用视差（1/depth），DA3 默认行为 |
| `percentile` | `2.0` | 使用 2% 和 98% 分位数，过滤极端值 |
| `cmap` | `"Spectral"` | colormap 名称，可选：turbo, viridis, inferno, jet 等 |
| `color_space` | `"linear"` | 线性或对数空间 |

---

## 三、配置参数（pandepth_config.yaml）

### 深度可视化相关配置

```yaml
# 可视化模式
depth_mode: colored  # "grayscale" 或 "colored"

# 彩色模式配置（仅 depth_mode="colored" 时生效）
depth_cmap: turbo              # colormap 名称
depth_percentile: 0.5          # 百分位数（注意：代码默认是 2.0）
use_disparity: true            # 是否使用视差（默认 true）
depth_color_space: linear     # "linear" 或 "log"

# 深度渲染类型（影响渲染时的深度计算方式）
depth_type: expected           # "expected" (ED) 或 "accumulated" (D)
```

**注意**：配置文件中 `depth_percentile: 0.5` 表示使用 0.5% 和 99.5% 分位数，比代码默认的 2.0% 更激进（过滤更多极端值）。

---

## 四、完整调用链路

### 渲染流程

```
1. 加载 checkpoint → 获取高斯参数（splats）
    ↓
2. 解析 COLMAP 数据 → 获取相机参数
    ↓
3. 对每个视角：
   a. render_cubemap_for_view()
      - 渲染 6 个 cubemap 面（RGB + Depth）
      - 使用 gsplat.rasterization()
      - depth_type: "expected" 或 "accumulated"
    ↓
   b. cubemap_to_equirect()
      - 将 cubemap 转换为全景图
    ↓
   c. create_cubemap_grid()
      - 为每个 cubemap 面调用 visualize_depth_map()
      - 生成 2 行 x 6 列的网格图（RGB + Depth）
    ↓
   d. create_pano_combined()
      - 为全景图调用 visualize_depth_map()
      - 生成上下拼接图（RGB 上，Depth 下）
```

### 可视化函数调用

```
visualize_depth_map()
    ↓
    ├─ mode="grayscale" → visualize_depth_map_gsplat_style()
    │                        └─ 简单线性归一化 → 灰度图
    │
    └─ mode="colored" → visualize_depth_map_colored()
                          ├─ use_disparity=True → 1/depth
                          ├─ color_space="log" → log(depth) [可选]
                          ├─ 百分位数归一化
                          ├─ 反转映射
                          └─ 应用 colormap → 彩色图
```

---

## 五、关键设计决策

### 1. 为什么默认使用视差（disparity）？

**问题**：深度值分布不均，大部分像素集中在某个区间，导致可视化几乎纯色。

**解决方案**：使用视差 `1/depth`
- 近处深度小 → 视差大
- 远处深度大 → 视差小
- **效果**：近景细节更明显，深度分布更均匀

**示例**：
```
原始深度：    [0.5, 1.0, 2.0, 10.0, 100.0]
转换为视差：  [2.0, 1.0, 0.5, 0.1,  0.01]
              ↑ 近处变化更明显
```

### 2. 为什么使用百分位数归一化？

**问题**：极端值（如天空、背景噪点）会压缩主要深度信息的对比度。

**解决方案**：使用百分位数（默认 2% 和 98%）
- 过滤掉最极端的 2% 最小值
- 过滤掉最极端的 2% 最大值
- **效果**：保留主要深度信息，提高对比度

### 3. 为什么需要反转映射？

**原因**：配合视差使用，确保颜色映射符合直觉
- 视差：近处值大，远处值小
- 反转后：近处 → 高值（红色），远处 → 低值（蓝色）
- **效果**：近=红，远=蓝，符合 DA3 风格

---

## 六、参数调优建议

### 如果深度可视化效果不理想：

1. **深度值被压缩到很小区间**
   - ✅ 已解决：默认启用 `use_disparity=True`
   - 检查：`depth_percentile` 是否过大（建议 0.5-2.0）

2. **近景细节不明显**
   - 尝试：`use_disparity=True`（默认已启用）
   - 尝试：`color_space="log"` 观察对数空间效果

3. **远景细节不明显**
   - 尝试：`use_disparity=False`（使用线性深度）
   - 尝试：调整 `depth_percentile`（如 1.0 或 0.5）

4. **颜色映射不理想**
   - 尝试不同的 `depth_cmap`：
     - `"turbo"`：高对比度，适合细节观察
     - `"Spectral"`：经典红-蓝映射
     - `"viridis"`：感知均匀
     - `"inferno"`：暗色背景，适合打印

5. **极端值影响可视化**
   - 减小 `depth_percentile`（如 0.5 或 1.0）
   - 检查渲染的深度值范围是否合理

---

## 七、调试信息

在 `create_cubemap_grid()` 中会输出详细的调试信息：

```
[Depth Debug] posz (raw depth): min=0.123, max=45.678, mean=12.345, std=8.901
[Depth Debug] posz percentiles: 1%=0.234, 5%=0.456, 10%=0.789, 90%=25.123, 95%=30.456, 99%=40.789
[Depth Debug] posz (disparity=1/depth): min=0.025, max=8.130, mean=0.123, std=0.456
```

**如何解读**：
- `raw depth`：原始深度值统计
- `disparity`：转换后的视差值统计（如果启用）
- `percentiles`：用于判断深度分布是否合理

---

## 八、总结

### 核心流程（彩色模式，默认）

```
原始深度 → 视差转换(1/depth) → 可选对数变换 → 百分位数归一化 → 反转映射 → colormap → 彩色图
```

### 关键参数

- ✅ `use_disparity=True`：默认启用，解决深度分布不均
- ✅ `percentile=2.0`：默认值，过滤极端值
- ✅ `cmap="Spectral"`：默认 colormap（可在配置中改为 "turbo"）
- ✅ `color_space="linear"`：默认线性空间（可选 "log"）

### 与 DA3 的一致性

- ✅ 默认使用视差（1/depth）
- ✅ 使用百分位数归一化
- ✅ 反转映射（近=红，远=蓝）
- ✅ 支持多种 colormap

---

**最后更新**：2024年（基于当前代码实现）
