# 출처 및 라이선스

## MatrAIx

이 저장소는 [MatrAIx](https://github.com/MatrAIx-ai/MatrAIx-Persona-8B)(commit 3633d8d)를 기반으로 한다. 실행할 때 MatrAIx 저장소의 속성 그래프(`persona/synthesis/graph/full_dag.json`), 속성 정의(`dimensions.json`), 샘플러와 템플릿을 불러와 사용한다. 이 저장소에 들어 있는 MatrAIx 유래 자료는 다음과 같다.

- `krpersona/render/dimensions_kr.json`: MatrAIx `dimensions.json`의 사본에 라벨 하나를 수정하고 새 한국 속성 21개를 더한 것.
- `krpersona/render/templating.patch`: MatrAIx `templating.py`의 import 한 줄을 바꾸는 패치.
- `krpersona/rules/catalogs/*.json`, `docs/tables/attributes.csv`: MatrAIx 속성의 id, 라벨, 분류, 설명, 값 목록을 담은 표.
- `krpersona/matraix/sampler.py`: MatrAIx 샘플러의 계산 방식을 따라 다시 사용한 수정 샘플러(차이는 `docs/07_생성.md`).

MatrAIx는 다음 MIT 라이선스로 배포된다.

```
MIT License

Copyright (c) 2026 MatrAIx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

MatrAIx를 인용할 때는 MatrAIx가 정한 다음 문헌을 인용한다.

```bibtex
@article{li2026matraix,
  title         = {MatrAIx: Simulating the World with 8.3 Billion Persona Agents},
  author        = {Li, Xiaomin and Hao, Yuexing and Hou, Jianheng and Huang, Jintao
                   and Wen, Qianfeng and Huang, Shirley and Liu, Yifan and Liu, Xiaoyi
                   and Fan, Yilan and Wang, Yijun and others},
  year          = {2026},
  eprint        = {2608.04205},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.04205}
}
```

## 자료

이 저장소에는 자료 제공처에서 받은 원자료 파일이 없다. 모형 추정과 검증에 사용한 자료는 다음과 같으며, 각 자료의 이용 조건에 따라 이용자가 제공 기관에서 직접 받아야 한다(`docs/03_자료.md`). 이 저장소의 결과는 제공 기관의 공식 통계가 아니며, 해석과 오류의 책임은 이 저장소에 있다.

| 자료 | 제공 |
|---|---|
| 2020 인구주택총조사 2% 표본(제공용) | 통계청(마이크로데이터 통합서비스) |
| 국민건강영양조사 2022(제9기 1차년도) | 질병관리청 |
| 가계금융복지조사 2025 | 통계청, 한국은행, 금융감독원(마이크로데이터 통합서비스) |
| 지역별고용조사 2025년 상반기 A형과 B형 | 통계청(마이크로데이터 통합서비스) |
| 한국종합사회조사(KGSS) 2003–2025 누적 자료 | 성균관대학교 동아시아학술원 서베이리서치센터 |
| 고령화연구패널(KLoSA) 10차 | 한국고용정보원 |
| OECD BFI-2-S 집계표(PIAAC 2주기) | OECD |

## 생성 입력 묶음

`bundle/`에는 추정 모형, 금지 규칙, 순차 CART 기반 가상 인구 기본 집단과 센서스 참조값을 포함한다. 건강 모형의 응답자 설계행렬은 제외하고 소득 금액은 구간별 가중 분위값 200개로 저장한다. 가족 모형의 비동거 자녀 연령 차이와 가중치는 응답자 5명 이상 묶음의 평균값이며, 센서스의 모자 나이 차이는 관측 순서를 제외한 빈도표로 저장한다.

이러한 저장 방식은 개별 기록의 직접 저장을 줄이기 위한 조치이며, 재식별 위험이 없거나 자료 제공처가 재배포를 허용한다는 보장은 아니다. 생성 입력 파일과 가상 인구 기본 집단의 공개 가능 여부는 아직 확인하지 않았다. 저장 형식과 파일 무결성 검사는 자료 제공처의 이용 조건 검토를 대신하지 않는다. 구체적인 구성은 `docs/10_한계.md` 10.13절을 참조한다.

## 문헌

- Choi, J., Kim, N., Zhang, B., Park, S. W., Cho, S., Sohn, Y. W., Soto, C. J., & John, O. P. (2026). The Big Five Inventory-2 in Korea: Validation and cross-cultural comparisons with the U.S. and Chinese versions. *Assessment*, 33(5), 866–889. https://doi.org/10.1177/10731911251357466
- 장승민. 한국판 IPIP-NEO-300 및 단축형 IPIP-NEO 척도의 개발 및 타당화. 2017년 선정 중견연구자지원사업 결과보고서(과제번호 2017S1A5A2A01027581). 성균관대학교 심리학과.
- Davenport, R. A., Krug, I., Rickerby, N., Dang, P. L., Forte, E., & Kiropoulos, L. (2024). Personality and cognitive factors implicated in depression and anxiety in multiple sclerosis: A systematic review and meta-analysis. *Journal of Affective Disorders Reports*, 17, 100832. https://doi.org/10.1016/j.jadr.2024.100832
- Gosling, S. D., Rentfrow, P. J., & Swann, W. B., Jr. (2003). A very brief measure of the Big-Five personality domains. *Journal of Research in Personality*, 37(6), 504–528.
- Parks-Leduc, L., Feldman, G., & Bardi, A. (2015). Personality traits and personal values: A meta-analysis. *Personality and Social Psychology Review*, 19(1), 3–29.

문헌의 값을 사용한 곳과 원문 대조 여부는 `docs/03_자료.md` 3.10절에 있다.
