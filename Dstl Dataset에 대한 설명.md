# Dstl: Dataset에 대한 설명

### 1. grid_sizes.csv

![image](https://user-images.githubusercontent.com/61573968/80793938-7bd2e300-8bd3-11ea-93cf-e2d8a5a18cc8.png)

이미지 ID, x의 max, y의 min의 순서임(0.0은 왼쪽 상단인 것 같음)



### 2. train_wkt_v4.csv

***WKT(Well-Known-Text)?***

```
점 point(x y)
선 linestring(x1 y1, x2 y2)
면 polygon((x1 y1, x2 y2, x3 y3, x4 y4))

멀티점 multipoint((x1 y1), (x2 y2), ...)
멀티선 multilinestring((x1 y1, x2, x2), (x1 y1, x2 y2))
멀티면 multipolygon(((x1 y1, x2 y2, x3 y3, x4 y4)), ((x1 y1, x2 y2, x3 y3, x4 y4)))
```

polygon의 괄호가 두 개인 이유? 속이 빈 polygon을 위해서



![image](https://user-images.githubusercontent.com/61573968/80794292-880b7000-8bd4-11ea-918e-8a1a8c2acc11.png)

이미지 ID, 이미지의  class type, multipoylgon 정보





