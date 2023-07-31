# 멀티모달 데이터 기반 추천 시스템 (Multi-modal Recommender System)

추천 시스템은 사용자의 정보를 분석하여 사용자에게 적합한 상품을 추천해주는 인공지능 기술 중 하나입니다. 추천 시스템 기술을 통해 사용자 편의성 증가 및 사용자의 상품의 접근성을 높여 기업의 이익 증대를 기대 할 수 있습니다.

추천 시스템은 주로 사용자의 상품에 대한 선호도 정보를 사용하지만, 데이터 수집의 어려움으로 Data Sparseness나 Cold Start 문제가 발생합니다. 이를 보완하고자, 최근 사용자 로그 정보 뿐만 아니라 이미지 혹은 리뷰 정보를 결합하여 Multi-modal 데이터 기반 추천 시스템 연구가 다수 진행되고 있습니다.

Multi-modal 데이터 기반의 고성능 추천 알고리즘 개발을 통해 추천 시스템의 한계를 극복하고 사용자에게 최적화된 개인화 추천 경험을 제공하는 것을 기대합니다.

## Index
* [대회 정보](#대회-정보)
* [모델](#모델)
* [코드 재현](#코드-재현)
***


## 대회 정보

- 주관: 인공지능융한연구센터, BK 산업융합형 차세대 인공지능 혁신인재 교육연구단
- 운영: 데이콘
- 대회 : [link](https://dacon.io/competitions/official/236113/overview/description)
- 대회 기간 : 2023.07.04 ~ 2023.08.07
- 평가 산식 : NDCG@50
    
    $DCG_u = \sum\limits_{l=1}^{50}\frac{relevance_i}{log2(i+1)}$

    $IDCG_u = \sum\limits_{l=1}^{50}\frac{relevance_i^{opt}}{log2(i+1)}$

    $NDCG_u = \frac{DCG_u}{IDCG_u}$

    $relevance_i$값은 평점이 3이상이면 1, 아니면 0으로 이진화 하여 계산

## 모델
- BM3 : [paper](https://arxiv.org/pdf/2207.05969.pdf) / [review]()

- Hyperparameter table : 5-fold를 통한 ndcg@50의 평균값을 기준으로 ndcg@50 내림차순 정렬

    | n_layers | embedding_size | feat_embed_dim | ndcg@50  | precision@50 | recall@50 | map@50   |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    |        4 |  256           |  128           | 0.036900 |     0.002700 |  0.093460 | 0.019720 |
    |        3 |  256           |  128           | 0.036800 |     0.002680 |  0.092940 | 0.019720 |
    |        4 |  128           |  128           | 0.036620 |     0.002740 |  0.094680 | 0.019100 |
    |        4 |  128           |  256           | 0.036600 |     0.002760 |  0.095020 | 0.019020 |
    |        4 |  128           |  64            | 0.036480 |     0.002740 |  0.094560 | 0.018980 |
    |        3 |  128           |  128           | 0.036300 |     0.002700 |  0.093700 | 0.018980 |
    |        3 |  128           |  256           | 0.036300 |     0.002700 |  0.093660 | 0.019000 |
    |        3 |  128           |  64            | 0.036240 |     0.002700 |  0.093280 | 0.019020 |
    |        6 |  128           |  128           | 0.036140 |     0.002740 |  0.094720 | 0.018580 |

<iframe src="https://wandb.ai/geonhyeock/inha-Competition/reports/Untitled-Report--Vmlldzo1MDE2NTc2" style="border:none;height:1024px;width:100%">

## 코드 재현

- [Raw data](data)

~~~ sh
# 모델 훈련 환경 구축
sh docker.sh
~~~

~~~python
# Data Preprocess
python preprocessing/preprocess.py

# Model Train
python src/main.py -m BM3 -d Inha

# Model Inference
python src/submission.py    

# submission 생성
cd ..
python src/ensemble.py -t weighted_voting
~~~
