# 멀티모달 데이터 기반 추천 시스템 (Multi-modal Recommender System)

- 🥇 대상 수상 - Winning Solution for a Competition 

추천 시스템은 사용자의 정보를 분석하여 사용자에게 적합한 상품을 추천해주는 인공지능 기술 중 하나입니다. 추천 시스템 기술을 통해 사용자 편의성 증가 및 사용자의 상품의 접근성을 높여 기업의 이익 증대를 기대 할 수 있습니다.

추천 시스템은 주로 사용자의 상품에 대한 선호도 정보를 사용하지만, 데이터 수집의 어려움으로 Data Sparseness나 Cold Start 문제가 발생합니다. 이를 보완하고자, 최근 사용자 로그 정보 뿐만 아니라 이미지 혹은 리뷰 정보를 결합하여 Multi-modal 데이터 기반 추천 시스템 연구가 다수 진행되고 있습니다.

Multi-modal 데이터 기반의 고성능 추천 알고리즘 개발을 통해 추천 시스템의 한계를 극복하고 사용자에게 최적화된 개인화 추천 경험을 제공하는 것을 기대합니다.



## Index
* [Competition imformation](#competition-imformation)
* [Data](#data)
* [Model](#model)
* [Result](#result)
* [Code reproduction](#code-reproduction)
***


## Competition imformation

- 주관: 인공지능융한연구센터, BK 산업융합형 차세대 인공지능 혁신인재 교육연구단
- 운영: 데이콘
- 대회 : [link](https://dacon.io/competitions/official/236113/overview/description)
- 대회 기간 : 2023.07.04 ~ 2023.08.07
- 평가 산식 : NDCG@50
    
    $DCG_u = \sum\limits_{l=1}^{50}\frac{relevance_i}{log2(i+1)}$

    $IDCG_u = \sum\limits_{l=1}^{50}\frac{relevance_i^{opt}}{log2(i+1)}$

    $NDCG_u = \frac{DCG_u}{IDCG_u}$

    $relevance_i$값은 평점이 3이상이면 1, 아니면 0으로 이진화 하여 계산

## Data

|name|count|
|:---:|:---:|
|user_id|192403|
|item_id|62989|
|interection|1254441|

item_id에 해당하는 image_feat, text_feat 제공

For more : [Raw data](data)


## Model
- BM3 : [paper](https://arxiv.org/pdf/2207.05969.pdf)

- Hyperparameter table
    - metric & inference_time : 5-fold average
    - Device : GeForce RTX 3080 Ti 12GB
    - ndcg@50 내림차순 정렬

    | n_layers | embedding_size | feat_embed_dim | ndcg@50  | precision@50 | recall@50 | map@50   | training_time_avg | inference_time_avg |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    |        4 |            256 |            128 | 0.036900 |     0.002700 |  0.093460 | 0.019720 | 4h 18m 0.60s      | 25.08s         |
    |        3 |            256 |            128 | 0.036800 |     0.002680 |  0.092940 | 0.019720 | 3h 55m 50.60s     | 20.56s         |
    |        4 |            128 |            128 | 0.036620 |     0.002740 |  0.094680 | 0.019100 | 3h 38m 19.20s     | 14.43s         |
    |        4 |            128 |            256 | 0.036600 |     0.002760 |  0.095020 | 0.019020 | 3h 20m 55.40s     | 14.37s         |
    |        4 |            128 |             64 | 0.036480 |     0.002740 |  0.094560 | 0.018980 | 3h 53m 47.20s     | 14.38s         |
    |        5 |            256 |            128 | 0.036380 |     0.002700 |  0.093560 | 0.019180 | 6h 33m 47.60s     | 29.33s         |
    |        3 |            128 |            128 | 0.036300 |     0.002700 |  0.093700 | 0.018980 | 4h 21m 45.40s     | 12.52s         |
    |        3 |            128 |            256 | 0.036300 |     0.002700 |  0.093660 | 0.019000 | 3h 17m 46.40s     | 12.46s         |
    |        3 |            128 |             64 | 0.036240 |     0.002700 |  0.093280 | 0.019020 | 3h 48m 49.60s     | 12.46s         |
    |        5 |            128 |            128 | 0.036140 |     0.002740 |  0.094780 | 0.018640 | 5h 33m 18.80s     | 16.31s         |
    |        6 |            128 |            128 | 0.036140 |     0.002740 |  0.094720 | 0.018580 | 4h 56m 59.20s     | 18.29s         |

- Drop_out : 0.5로 고정

    [Table Visualization](https://api.wandb.ai/links/geonhyeock/8vz3j6ru)

## Result

- best5 parameter model ensemble : 25개의 csv 파일 (model : 5 and fold : 5)
- Hard_voting : 각 모델이 유저별 예측한 아이템의 빈도수를 기준으로 큰 값부터 추천
- weighted_voting : Hard_voting에서 $i$ 번째 등장한 아이템에 대하여 $\frac{1}{log_2(i+1)}$ 가중치를 더하여 큰 값부터 추천


|Type|Public(30%)|Private|
| :---: | :---: | :---: |
| weighted_voting | 0.0428 | 0.0442 |
| Hard_voting     | 0.0386 | 0.0399 |


## Code reproduction

- [데이터](data)
- config : [model](MMRec/src/configs/model), [dataset](MMRec/src/configs/dataset), [overall](MMRec/src/configs/)

~~~ sh
# 모델 훈련 환경 구축
# docker는 CUDA Version: 11.2 기준으로 작성되었습니다. Dockerfile을 확인해주세요
sh docker.sh
~~~

~~~python
# Data Preprocess
python preprocessing/preprocess.py

# Model Train
python src/main.py -m BM3

# Model Inference
python src/submission.py

# submission 생성
cd ..
python src/ensemble.py -t weighted_voting -f BM3
~~~

~~~
앙상블 결과 경로 : /workspace/root/Challenge-Multi-modal-Recommender-System/submission/best.csv
docker cp [container ID]:[앙상블 결과 경로] [host 파일경로]
~~~

## [Report](report.pdf)

