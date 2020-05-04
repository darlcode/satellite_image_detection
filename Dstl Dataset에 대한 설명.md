# Dstl: Dataset에 대한 설명

```
Dstl은 3 밴드 및 16 밴드 형식의 1km x 1km 위성 이미지

좌표의 이해가 통제되는 것처럼 보입니다. 
이미지 픽셀이 얼마나 큰지 확인하기 위해 할 수있는 또 다른 방법은 특정 데이터의 인용 해상도를 보는 것입니다. 
~ 0.3m 해상도는 1km 거리의 이미지에서 ~ 3333 픽셀을 의미합니다. 
거친 해상도, 즉 10m는 가로로 100 픽셀 만있는 이미지를 의미합니다. 
다른 센서에서 나오는 다양한 이미지의 크기 차이를 설명합니다.
```

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



xmax, ymin:

[참고 링크1]( https://www.kaggle.com/lorismichel/exploration-of-nearby-images )

[참고 링크2]( https://www.kaggle.com/lorismichel/number-of-xmax-and-ymin-variants/code )

