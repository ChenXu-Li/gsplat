# 深度归一化范围：Cubemap Grid vs Panorama

## 问题

在深度可视化时，最大深度值和最小深度值的计算是：
- **每个 cubemap 视角单独计算**（每个面独立归一化）？
- **还是统一计算**（所有面一起归一化）？

## 答案

**答案取决于输出类型**：

1. **Cubemap Grid** (`{view_tag}_cubemap_grid.png`): **每个面单独计算**
2. **Panorama Combined** (`{view_tag}_pano_combined.png`): **统一计算**

---

## 详细分析

### 1. Cubemap Grid - 每个面单独计算

**函数**: `create_cubemap_grid()`

**代码逻辑**:
```python
# Second row: Depth images
for col, face_name in enumerate(face_order):
    if face_name in cube_depths:
        depth_face = cube_depths[face_name]  # 单个面的深度图
        
        # 对每个面单独调用 visualize_depth_map
        depth_vis = visualize_depth_map(
            depth_face,  # 只传入单个面的深度图
            mode=depth_mode,
            cmap=depth_cmap,
            percentile=depth_percentile,
            use_disparity=use_disparity,
            color_space=color_space,
            debug_label=f"cubemap_{face_name}",
        )
        grid[face_h:2*face_h, col*face_w:(col+1)*face_w, :] = depth_vis
```

**特点**:
- ✅ 每个 cubemap 面（posx, negx, posy, negy, posz, negz）**独立计算** min/max
- ✅ 每个面使用自己的深度范围进行归一化
- ✅ 不同面的颜色映射可能不一致（因为归一化范围不同）

**示例**:
```
posz 面: 深度范围 [0.1, 10.0] → 归一化到 [0, 1] → 灰度 [0, 255]
posx 面: 深度范围 [0.5, 50.0] → 归一化到 [0, 1] → 灰度 [0, 255]
negy 面: 深度范围 [1.0, 100.0] → 归一化到 [0, 1] → 灰度 [0, 255]
```
- 虽然深度范围不同，但每个面都映射到完整的 [0, 255] 范围
- 相同深度值在不同面可能显示为不同的灰度值

**优点**:
- ✅ 每个面都能充分利用颜色/灰度范围
- ✅ 每个面的细节都能清晰显示

**缺点**:
- ❌ 不同面之间的颜色/灰度值不能直接比较
- ❌ 相同深度值在不同面可能显示为不同颜色

---

### 2. Panorama Combined - 统一计算

**函数**: `create_pano_combined()`

**代码逻辑**:
```python
# Convert depth to visualization
depth_vis = visualize_depth_map(
    pano_depth,  # 传入整个全景深度图 [H, W]
    mode=depth_mode,
    cmap=depth_cmap,
    percentile=depth_percentile,
    use_disparity=use_disparity,
    color_space=color_space,
    debug_label="pano_combined",
)
```

**特点**:
- ✅ 对整个全景深度图**统一计算** min/max
- ✅ 所有像素使用相同的深度范围进行归一化
- ✅ 整个全景图的颜色映射一致

**示例**:
```
整个全景图: 深度范围 [0.1, 100.0] → 归一化到 [0, 1] → 灰度 [0, 255]
```
- 所有像素都基于这个统一范围进行归一化
- 相同深度值在整个全景图中显示为相同的灰度值

**优点**:
- ✅ 整个全景图的颜色/灰度值可以相互比较
- ✅ 相同深度值显示为相同颜色/灰度
- ✅ 更符合深度值的物理意义

**缺点**:
- ❌ 如果深度范围很大，某些区域的细节可能不够明显
- ❌ 深度范围小的区域可能只使用部分颜色/灰度范围

---

## 可视化函数内部实现

### Grayscale 模式 (`visualize_depth_map_gsplat_style`)

```python
# 计算 min/max（基于传入的深度图）
depth_min = depth_vis[valid_mask].min()  # 只考虑当前深度图
depth_max = depth_vis[valid_mask].max()  # 只考虑当前深度图

# 归一化
depth_norm = (depth_vis - depth_min) / (depth_max - depth_min)
```

**关键点**: 
- min/max 只基于**传入的深度图**计算
- 如果传入单个面 → 基于该面计算
- 如果传入全景图 → 基于整个全景图计算

### Colored 模式 (`visualize_depth_map_colored`)

```python
# 计算百分位数 min/max（基于传入的深度图）
depth_min = np.percentile(valid_depth, percentile)  # 只考虑当前深度图
depth_max = np.percentile(valid_depth, 100.0 - percentile)  # 只考虑当前深度图

# 归一化
depth_norm = (depth - depth_min) / (depth_max - depth_min)
```

**关键点**: 
- 百分位数 min/max 只基于**传入的深度图**计算
- 如果传入单个面 → 基于该面计算
- 如果传入全景图 → 基于整个全景图计算

---

## 实际影响

### 场景 1: 不同方向的深度范围差异很大

**示例**: 
- 前方（posz）: 深度范围 [0.1, 5.0]
- 后方（negz）: 深度范围 [10.0, 100.0]

**Cubemap Grid**:
- posz 面: 使用 [0.1, 5.0] 归一化 → 细节清晰
- negz 面: 使用 [10.0, 100.0] 归一化 → 细节清晰
- 但两个面的颜色不能直接比较

**Panorama Combined**:
- 整个全景图: 使用 [0.1, 100.0] 归一化
- posz 区域: 深度值 [0.1, 5.0] → 映射到 [0, 0.05] → 灰度 [0, 13]（很暗）
- negz 区域: 深度值 [10.0, 100.0] → 映射到 [0.1, 1.0] → 灰度 [26, 255]（较亮）
- 前方细节可能不够明显（只使用了很小的灰度范围）

### 场景 2: 深度范围相对均匀

**示例**: 
- 所有方向: 深度范围大致在 [1.0, 50.0]

**Cubemap Grid**:
- 每个面都使用自己的范围归一化
- 细节清晰，但颜色略有差异

**Panorama Combined**:
- 整个全景图使用统一范围归一化
- 颜色一致，细节也清晰

---

## 代码位置

### Cubemap Grid 实现
**文件**: `code/gsplat/lcxscripts/pandepth_exporter.py`
**函数**: `create_cubemap_grid()` (第 843-946 行)
**关键代码**: 第 888-944 行
```python
for col, face_name in enumerate(face_order):
    depth_face = cube_depths[face_name]  # 单个面
    depth_vis = visualize_depth_map(depth_face, ...)  # 单独处理
```

### Panorama Combined 实现
**文件**: `code/gsplat/lcxscripts/pandepth_exporter.py`
**函数**: `create_pano_combined()` (第 949-999 行)
**关键代码**: 第 984-992 行
```python
depth_vis = visualize_depth_map(pano_depth, ...)  # 整个全景图
```

---

## 总结

| 输出类型 | 归一化范围 | 计算方式 | 颜色一致性 |
|---------|-----------|---------|-----------|
| **Cubemap Grid** | 每个面独立 | 单独计算 | ❌ 不一致（每个面独立归一化） |
| **Panorama Combined** | 整个全景图统一 | 统一计算 | ✅ 一致（整个全景图统一归一化） |

**建议**:
- 如果需要**比较不同方向的深度值** → 使用 Panorama Combined
- 如果需要**每个方向都显示清晰细节** → 使用 Cubemap Grid
- 如果需要**两者兼顾** → 可以同时查看两种输出

---

## 潜在改进

如果需要让 Cubemap Grid 也使用统一归一化，可以修改 `create_cubemap_grid()` 函数：

```python
# 先计算所有面的全局 min/max
all_depths = np.concatenate([cube_depths[name][cube_depths[name] > 0] 
                             for name in face_order])
global_min = np.percentile(all_depths, percentile)
global_max = np.percentile(all_depths, 100.0 - percentile)

# 然后对每个面使用全局范围归一化
for col, face_name in enumerate(face_order):
    depth_face = cube_depths[face_name]
    depth_vis = visualize_depth_map_with_range(
        depth_face, 
        global_min=global_min, 
        global_max=global_max,
        ...
    )
```

这样可以保证 Cubemap Grid 中所有面的颜色映射一致。
