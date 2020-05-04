# Full pipeline demo: poly -> pixel -> ML -> poly

[본 포스팅 주소]( https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly )

이 스크립트는 픽셀 기반 분류기(pixel-based classifier)의 전체 training 및 prediction 파이프 라인을 보여줌

마스크를 만들고, 1 픽셀 패치(one-pixel patches)에서 로지스틱 회귀를 훈련하고, 모든 픽셀을 예측하고, 픽셀에서 다각형을 만들고 매끄럽게 만듦



1. 필요한 라이브러리를 import 함

```python
from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

csv.field_size_limit(sys.maxsize);
```



2. 이미지 6120_2_2에서 building(class 1)에 대해 작업을 함, 우선 gide size와 polygon들을 로드함

```python
IM_ID = '6120_2_2'
POLY_TYPE = '1'  # buildings
```

- grid_sizes.csv파일을 읽어 gird의 x_max, y_min값을 로드

```python
# Load grid size
x_max = y_min = None
for _im_id, _x, _y in csv.reader(open('../Dataset/Dstl_kaggle/grid_sizes.csv')):
    if _im_id == IM_ID:
        x_max, y_min = float(_x), float(_y)
        break
```

- train_wkt_v4.csv 파일을 읽어 polygon의 정보를 로드

```python
# Load train poly with shapely
train_polygons = None
for _im_id, _poly_type, _poly in csv.reader(open('../Dataset/Dstl_kaggle/train_wkt_v4.csv')):
    if _im_id == IM_ID and _poly_type == POLY_TYPE:
        train_polygons = shapely.wkt.loads(_poly)
        break
```

```python
print(train_polygons)
##출력 결과
#MULTIPOLYGON (((0.000439 -0.009039999999999999, 0.000438 -0.008999, 0.000637 -0.008985999999999999, 0.000644 -0.009039999999999999, 0.000439 -0.009039999999999999)), ((0.008607999999999999 -0.009039999999999999, 0.008564 -0.008978, 0.008666 -0.008906000000000001, 0.008770999999999999 -0.009039999999999999, ...
```

- tiff를 통해 tif파일을 읽음

```python
# Read image with tiff
im_rgb = tiff.imread('../Dataset/Dstl_kaggle/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
im_size = im_rgb.shape[:2]
```

```python
print('../Dataset/Dstl_kaggle/three_band/{}.tif'.format(IM_ID))
## 출력결과
#../Dataset/Dstl_kaggle/three_band/6120_2_2.tif
print(im_size)
## 출력결과
# (3348, 3403)
```

```python
tiff.imshow(im_rgb)
```

<img src="https://user-images.githubusercontent.com/61573968/80912319-220b1e00-8d77-11ea-873b-a83d692942b0.png" alt="image" style="zoom:67%;" />



- 이미지와 grid의 비를 조정하기 위해 polygon scaler를 구하는 함수

```python
def get_scalers():
    h, w = im_size # mask_for_polygons가 제대로 작동하도록 뒤집음
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

x_scaler, y_scaler = get_scalers()

train_polygons_scaled = shapely.affinity.scale(
    train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

```

` shapely.affinity.scale(geom, xfact=1.0, yfact=1.0, zfact=1.0, origin='center')`

- polygons을 이용하여 mask를 만드는 함수정의

```python
def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

train_mask = mask_for_polygons(train_polygons_scaled)

```

` uint8`: 양수, $$ 2^8 $$개수 만큼 표현 가능, 0 ~ 255

`int32`: $$ 2^{32} $$개수 만큼 표현 가능, -2,147,483,648 ~ 2,147,483,647

- 더 나은 display를 위한 함수

```python
def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

```

[np.percentile 설명]( https://docs.scipy.org/doc/numpy/reference/generated/numpy.percentile.html )

- 이미지와 마스크 출력

```python
tiff.imshow(255 * scale_percentile(im_rgb[2900:3200,2000:2300]));

```

<img src="https://user-images.githubusercontent.com/61573968/80913903-14f42c00-8d83-11ea-9c99-d2d61727629b.png" alt="image" style="zoom:67%;" />

```python
def show_mask(m):
    # hack for nice display
    tiff.imshow(255 * np.stack([m, m, m]));
show_mask(train_mask[2900:3200,2000:2300])

```

<img src="https://user-images.githubusercontent.com/61573968/80913942-44a33400-8d83-11ea-8ea8-7251932a6d39.png" alt="image" style="zoom:67%;" />
