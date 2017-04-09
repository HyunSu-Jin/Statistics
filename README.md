# statistics_simple
데이터 사이언스, 데이터마이닝에서 요구되는 통계학적 기본지식들을 python 구현과 함께 나타내었다.

1.데이터 시각화
- matplotlib

![img1](/img/img1.png)

![img2](/img/img2.png)

2. 평균(mean), 중앙값(median), 분위(quantile), 최빈값(mode)
- 평균은 데이터가 나태는 값들의 전체 합을 데이터의 수로 나눈것을 의미한다.
- 중앙값은 전체 데이터를 정렬하였을때 정 가운데 인덱스가 나타내는 값을 의미한다
- 분위,p가 주어졌을 때. 이에대한 분위는 데이터를 정렬하고 데이터의 수 * p 에 해당하는 인덱스가 나타내는 값을 의미한다.
- 최빈값은 데이터가 나타내는 값들 중 가장 빈번하게 나타난 값을 의미한다.

평균은 데이터셋에 존재하는 각각의 값의 변화에 민감하다. 예를들어, 데이터셋의 최댓값이 100에서 10000으로 변경되었다면 평균값은 변경되나 중앙값은 변동하지 않는다또한, 평균값은 Noise에 대해서 상당히 민감한데, 상위2%가 전체 데이터의 평균을 dominant할 수 있다는 뜻이다. 이에 대한 해결책으로 데이터셋에대한 평균값을 구하고자 할 때 상위2%, 하위2%에 해당하는 데이터를 무시하고 나머지 데이터만을 통해 평균값을 구하는 방법이 있다.

<pre><code>
def median(data):
    n = len(data)
    sorted_v = sorted(data)
    middle = n // 2
    if n % 2 == 1:
        return data[middle]
    else:
        left = middle-1
        right = middle
        return (data[left] + data[right]) / 2
</code></pre>

<pre><code>
def quantile(x,p):
    p_index = int(len(x) * p)
    return sorted(x)[p_index]
</code></pre>

<pre><code>
def mode(x):
    counts = Counter(x) # key : 값 , value : count
    #print(counts)
    max_count = max(counts.values())
    #print(max_count)
    return [ key for key,count in counts.items() if count == max_count]
</code></pre>


3. 산포도, 분산, 표준편차

- 산포도(dispersion)은 데이터가 얼마나 퍼져 있는지를 나타낸다. 데이터가 모두 같은값을 가진 경우와 데이터간 값의 차이가 큰 경우를 구분짓기 위함이다.
<pre><code>
def dispersion(vec):
    return max(vec)- min(vec)
</code></pre>
이에 대한 예로써 위와같이 데이터셋의 maximum과 minimum간의 차이로 dispersion한 정도를 나타낼 수 있겠으나, 위에 대한 지표는 [0,100] 과 [0,100]에 50 이라는 값이 100개 추가된 vector간의 차이를 구분지을 수 없다. 이를 구분짓기 위하여 다음과 같이 분산, 표준편차를 이용한다.

- 분산(variance)
분산은 데이터셋에서 각 data에 평균값을 뺀뒤 제곱한값의 평균이다. 이 지표는 데이터셋이 얼마나 dispersion되어 있느냐를 해당 데이터의 단위제곱으로 나태낸다.
<pre><code>
def variance(vec):
    vec=  np.array(vec)
    vec = vec - mean(vec)
    vec = vec ** 2
    return np.sum(vec) / (vec.shape[0]-1)
</code></pre>

- 표준편차
표준편차는 데이터셋의 각 데이터에 평균갑을 뺀값의 평균을 나타낸다. 이 값은 분산의 제곱근이다.

분산과 표준편차는 평균값과 같이 Noise 데이터에 민감하게 변동되므로 다음과 같이 분위를 이용한 방법으로 dispersion 정도를 나타내기도 한다.
<pre><code>
def interquartile(vec):
    return quantile(vec,0.75) - quantile(vec,0.25)
</code></pre>
