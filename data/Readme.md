# Dataset Info.
코드 재현을 위해서는 [raw](/raw) 폴더에 데이터를 다운받아주세요.
- [download-link](https://dacon.io/competitions/official/236113/data)



### train.csv [파일]
user_id : 유저 고유 식별 ID\
item_id : 아이템 고유 식별 ID\
rating : 해당 유저가 대상 아이템에 부여한 평점 (1, 2, 3, 4, 5점)

---


### image.npy [파일]
item_id와 매핑되는 item의 이미지 feature 데이터 (I, 4096)\
image_1.npy ~ image_4.npy는 image.npy 로드시 memory overflow가 나는 경우를 위해, 20000개 단위로 Split한 파일

---

### text.npy [파일]
item_id와 매핑되는 item의 리뷰 feature 데이터 (I, 384)

---

### user_label.npy & item_label.npy
분석을 위한 라벨 관련 정보 (정답 라벨의 라벨 의미가 아님)

---

제출 파일 형식 (sample_submission.csv 별도 제공하지 않음)\
train.csv에 존재하는 모든 user_id 별 추천 아이템 리스트 우선 순위 상위 50개 순서로 예측 (50개 미만 시 제출 오류 처리)\
컬럼이 2개(user_id, item_id)인 csv 형태로 제출
베이스라인 코드로부터 생성된 결과(submit.csv)를 반드시 참고


