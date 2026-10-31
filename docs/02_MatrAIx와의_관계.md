# 2. MatrAIx와의 관계

kr-persona는 MatrAIx의 속성 체계와 확률 그래프 위에 한국 자료로 생성한 값을 얹는 방식으로 페르소나를 생성한다. MatrAIx의 원본 구조와 한국 성인 표본에서 확인한 문제, kr-persona의 변경 범위를 설명한다.

## 2.1 MatrAIx

MatrAIx(https://github.com/MatrAIx-ai/MatrAIx-Persona-8B)는 페르소나를 LLM 에이전트로 생성하여 설문, 챗봇, 웹, 앱 네 환경의 과제를 수행하게 하는 평가 기반 시설이다. 페르소나는 범주형 속성의 묶음이며, MatrAIx는 이를 그래프 기반 합성 생성과 사람 자료에 근거한 기록으로 생성한다. 이 저장소가 사용하는 부분은 속성 정의(`persona/schema/`), 합성 그래프와 샘플러(`persona/synthesis/`), 공개 자료 형식(`persona/post_process/unified_dataset/`), 페르소나 렌더러(`src/matraix/persona_dimension_catalog.py`, `environment/agents/matraix/agents/persona/templating.py`)이고, 과제와 실행 환경은 사용하지 않는다.

기준 판은 커밋 `3633d8dab149a9482a71b024418a49ae828cc941`(2026-09-09)이며, 그래프 판은 `v4.4_developer_extension`이다. MatrAIx는 저장소에 포함하지 않는다. 사용자가 같은 커밋을 따로 받아 두고 `config/paths.toml`의 `matraix_dir`(또는 환경 변수 `KRP_MATRAIX`)로 가리킨다. `krpersona/paths.py`의 `MATRAIX_COMMIT = '3633d8d'`은 기록용 상수이며, 실행 중에 checkout의 커밋을 확인하지는 않는다.

### 속성과 분류

`persona/schema/dimensions.json`은 속성 1,290개를 정의한다. 속성마다 id, 이름(label), 분류(category), 설명, 값 목록, 기본값(`defaultValue`), 문장 틀(`phrase`)이 있다. 값의 수는 2\~16개(중앙값 5, 합계 6,347개)다. 모든 페르소나는 속성마다 값을 정확히 하나 가진다. 분류는 43개이며, 파일의 설명에 따르면 화면에서 묶어 보여 주기 위한 표지일 뿐 속성 사이의 위계를 뜻하지 않는다. 속성이 많은 분류는 다음과 같다.

| 분류 | 속성 수 | 분류 | 속성 수 |
|---|---|---|---|
| Expertise: Domains | 144 | Linguistic: Language | 53 |
| Interests: Media | 81 | Professional: Industry | 51 |
| Interests: Topics | 78 | Personality: Big Five | 50 |
| Interests: Culture | 74 | Interests: Hobbies | 50 |
| Skills: Tools | 69 | Values & Motivation | 46 |
| Worldview: Beliefs | 67 | Skills: Programming | 44 |
| Expertise: Skills | 64 | 그 밖 30개 분류 | 469 |

1,290개 가운데 1,126개는 `defaultValue`가 있다(예: 성격 하위 요인은 Average, `gender_identity`는 Prefer not to say). 이 값은 아래 렌더러에서 "정보 없음"으로 취급된다. 이 저장소가 출력하는 1,311개는 이 1,290개에 새 한국 속성 21개를 더한 것이다(4.4절).

### 그래프

그래프는 `persona/synthesis/graph/full_dag.json`(24.8 MB) 하나다. 이 문서의 개수는 모두 파일의 배열에서 직접 집계한 값이다. 파일의 `metadata`에 적힌 개수 가운데 일부(예: `conditional_mask_count` 267)는 실제 배열(524개)과 맞지 않는다.

| 구성 요소 | 개수 | 내용 |
|---|---|---|
| 노드 | 1,308 | 출력 속성 1,290 + 잠재 노드 18(`emit: false`) |
| 쌍별 연결 | 6,999 | 부모 값별 자식 분포 $P_e(v \mid u)$와 가중치 $w_e$ |
| 전체 조건부 확률표(CPT) | 54 | 부모 2\~3개의 값 조합별 자식 분포, 17,645행 |
| 조건부 금지 규칙(마스크) | 524 | 조건이 맞으면 지정한 값의 확률에 0을 곱함 |
| 위상 순서 | 1,308 | `proposal_view.topological_order` |

각 노드에는 기본 확률 $P^0_i$가 있다. 그래프 메타데이터는 목표 모집단을 "세계 일반 인구"로 적고 있다. 부모가 없는 뿌리 노드는 86개(출력 속성 68개, 잠재 노드 18개)다. 잠재 노드 18개는 여러 속성을 함께 움직이게 하려고 둔 노드로, 값은 Low, Medium, High이고 기본 확률은 0.25, 0.5, 0.25이며 출력되지 않는다.

쌍별 연결 6,999개는 노드 쌍 854,778개의 0.82%만 잇는다. 연결 가중치는 0.04\~0.95(중앙값 0.378, 평균 0.455)다. 연결마다 붙은 근거 표지는 다음과 같다.

| 근거 표지(`evidence_level`) | 연결 수 | 비율 |
|---|---|---|
| `schema_semantic_prior`(속성 정의에 기댄 사전값) | 5,072 | 72.5% |
| `manual_structural_prior`(수작업 구조 사전값) | 550 | 7.9% |
| `raw_direct`(원자료에서 직접 추정) | 107 | 1.5% |
| `weak_heuristic` | 1 | 0.0% |
| 표지 없음 | 1,269 | 18.1% |

CPT 54개의 출처 표지는 `v4_production_repair` 21개, `stackoverflow_survey_2025` 17개, `v3_global_all_humans_repair` 6개, `wvs_wave7_csv_v6_0` 3개, `acs_pums_2024_1yr` 1개, `persona8b_age_generation_mapping_2026` 1개, 표지 없음 5개다. 54개 모두 `replace_pairwise_parent_edges = true`이므로, CPT의 부모에서 같은 자식으로 가는 쌍별 연결은 계산에서 제외된다.

마스크 524개는 모두 `bad_values`와 `bad_value_multiplier = 0`으로 되어 있고, `downweight_values`나 `penalize_values_outside_preferred_set`을 사용하는 마스크는 없다. `constraint_semantics` 표지는 36종이며(`soft_compatibility_language_budget` 100개, `hard_guard` 90개 등) 이름에 soft가 붙은 것도 많지만, 곱하는 값이 모두 0이므로 샘플러에서는 모두 절대 금지로 작동한다. 마스크의 조건은 조건 속성마다 허용 값 집합을 적은 것이며, 모든 조건 속성이 동시에 맞을 때만 작동한다.

### 샘플러

`persona/synthesis/sampler/sampler.py`의 `PersonaForwardSampler`는 위상 순서대로 노드를 하나씩 추출한다. 노드 $i$의 값 $v$는 다음 확률로 추출된다.

$$
\log q_i(v) = \log P^0_i(v) + \gamma_i\Big[\sum_{e \in E_i} w_e\big(\log P_e(v \mid x_{s(e)}) - \log P^0_i(v)\big) + \sum_{c \in C_i} w_c\big(\log P_c(v \mid \mathbf x_{\mathrm{pa}(c)}) - \log P^0_i(v)\big)\Big] + \sum_{m \in M_i} \log \mu_m(v \mid \mathbf x) + \text{상수}
$$

$$
\gamma_i = \frac{1}{\max\big(1, \sqrt{\sum_{c \in C_i} w_c^2 + \sum_{e \in E_i} w_e^2}\big)}, \qquad \mu_m(v \mid \mathbf x) = \begin{cases} 0 & \text{조건이 맞고 } v \in \text{금지 값} \\ 1 & \text{그 밖} \end{cases}
$$

$E_i$는 $i$로 들어오는 쌍별 연결 가운데 CPT 부모에서 오는 것을 뺀 집합, $C_i$는 $i$를 자식으로 하는 CPT, $M_i$는 $i$를 대상으로 하는 마스크다. 확률은 $10^{-12}$ 아래로 내려가지 않게 자른다. $\gamma_i$는 부모가 많은 노드의 분포가 지나치게 뾰족해지지 않게 하는 수축 계수다. 값은 정규화하지 않은 누적 분포에서 역변환으로 추출하며, 난수는 `numpy.random.default_rng(seed)` 하나에서 나온다. 마스크를 곱한 뒤 모든 값의 확률이 0이 된 행은 균등 분포로 복원한다. `sample_indices(n, fixed=...)`로 일부 노드를 모든 행에서 같은 값으로 고정할 수 있지만, 행마다 다른 값을 넣는 기능은 없다.

### 렌더러와 공개 형식

렌더러(`persona_dimension_catalog.py`)는 속성을 분류에 따라 열 개 절(Identity, Career & education, Language & communication, Personality & values, Current interaction state, Worldview, Interests, Skills & expertise, Lifestyle & health, Developer & AI)로 묶고, 속성마다 "- 이름: 값" 한 줄을 출력한다. 이때 값이 비었거나 `none`, `not applicable`, `not a developer`, `no interest`, `prefer not to say` 같은 목록에 있거나 속성의 `defaultValue`와 같으면 그 줄을 사용하지 않는다. 글자 수 상한을 줄 수 있지만 기본값은 상한 없음이다. `templating.py`가 이 결과를 `persona_system.md.j2` 틀에 넣어 에이전트의 시스템 프롬프트를 생성한다.

샘플러의 기본 출력은 `persona_codes` 형식이다. 값이 모두 16개 이하이므로 속성 하나를 4비트로 적어 한 행이 645바이트이고, 열 목록과 값 목록은 `<파일>.schema.json`에 따로 기록한다. Hugging Face에 공개한 Persona 1M(`MatrAIx2026/MatrAIx_Persona_1M_Public_Release`)은 `persona/post_process/unified_dataset/schema.py`의 `UNIFIED_SCHEMA` 형식 parquet이다. 열은 `source`, `source_row_index`, `source_record_id`, `attributes`(1,290개 4비트 코드, 645바이트 고정), `null_bitmap`(162바이트), `attribute_overrides`(값 목록 밖의 값: 속성 번호와 문자열), `has_description`, `descriptions`, `grounding`, `metadata_json`이다. 코드와 문자열 사이의 변환은 `AttributeCodec`이 한다. 공개 전 품질 거르기 규칙은 `persona/post_process/quality_filter/contradictions.json`(22개, 조건과 대상 값이 모두 맞으면 행을 버림)에 있다.

### 인용

MatrAIx는 README와 `CITATION.cff`에서 다음 논문을 인용하도록 요청한다.

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

전체 저자 목록은 MatrAIx 저장소의 `CITATION.cff`에 있다.

## 2.2 MatrAIx를 한국 성인에 그대로 사용할 때의 문제

### 비교 표본과 점검 방법

비교에는 공개 Persona 1M 파일 대신 MatrAIx의 공식 그래프와 샘플러로 재생성한 표본을 사용했다. 코드는 `krpersona/baseline/`에 있고, 결과는 산출물 폴더의 `artifacts/baseline/`에 있다. 실행 순서는 `scripts/01_baseline.sh`에 있다.

1. `baseline/gen.py`: `PersonaForwardSampler(DEFAULT_GRAPH_PATH, SamplingConfig(seed=20261014))`로 10만 명씩 10번, 모두 100만 명을 고정 없이 추출한다. MatrAIx의 `contradictions.json` 규칙 22개를 같은 방식(조건과 대상 값이 모두 맞으면 버림)으로 적용하면 44명이 제외되고 999,956명이 남는다. 이 가운데 `cult_south_korea = Native`인 52,864명(5.29%)을 "한국 문화 Native" 집단으로, 묶음마다 5,000명씩 균등 비복원추출한 50,000명(시드 7)을 "전체" 비교 집단으로 저장한다(`codes.npz`, `meta.json`, `gen_stats.json`). 전체 비교 집단에도 한국 문화 Native가 2,578명 들어 있다.
2. `baseline/analyze.py`: 한국인으로서의 그럴듯함과 성격과 가치의 내부 일관성을 보는 지표 41개와 주요 속성 14개의 분포(`results.json`).
3. `baseline/demo_checks.py`: 한 페르소나 안의 모순 규칙 43개(hard 33개, soft 10개; `demo_results.json`).
4. MatrAIx의 `persona/synthesis/scripts/generate_quality_report.py`를 기본값(10,000명, 시드 42)으로 그대로 실행한 결과(`qr/`). MatrAIx checkout에서 직접 실행했으며 `scripts/01_baseline.sh`에는 들어 있지 않다.

MatrAIx에서 "한국 문화 Native"를 고르는 방법은 `cult_south_korea` 하나뿐이다. 이 노드는 `region`에서 오는 연결 하나(가중치 0.3194)만 받고, 다른 노드로 나가는 연결이 없다. 따라서 이 값으로 선택한 집단은 `region`이 모두 East Asia(100%)라는 점을 빼면 다른 속성에 대해 East Asia 페르소나 일반과 다르지 않다. 한국 문화 Native 52,864명 가운데 15,627명(29.6%)은 18세 미만이며, 성인 37,237명(MatrAIx 연령대 18-24 이상)이 뒤의 검증(9장)에서 MatrAIx 비교 표본으로 사용된다. MatrAIx에는 만 나이가 없으므로 18-24에는 18세가 들어 있다.

### 한국인으로서의 그럴듯함

아래 수치는 모든 연령을 포함한 52,864명(한국)과 50,000명(전체) 기준이다(`results.json`). 성인 37,237명으로 한정한 값은 괄호에 적었다.

| 점검 | 한국 문화 Native | 전체 |
|---|---|---|
| 한국어 Native 또는 Fluent | 0.0% | 0.0% |
| 한국어 None | 66.0%(성인 65.9%) | 65.9% |
| 한국어 None 또는 Basic | 96.1% | 95.8% |
| 주 언어 Mandarin | 82.4%(성인 82.4%) | 20.4% |
| 중국어 Native | 55.5% | 14.0% |
| 중국 또는 일본 문화도 Native | 46.9% | 9.1% |
| `demo_ethnicity_broad` East Asian | 91.7% | 20.6% |
| `cultural_background` Collectivist (East Asian) | 12.6% | 12.6% |
| 문화 배경이 동아시아, 혼합이 아님 | 75.0% | 75.1% |
| 출생 시민이면서 이민 1세대 | 17.4% | 17.2% |
| 한국 음식을 피함(Avoid) | 22.6% | 22.9% |
| 종교 Christian | 6.9%(성인 7.0%) | 29.0% |
| 65세 이상 | 10.1%(성인 중 14.3%) | — |

`primary_language`의 값 목록에는 Korean이 없고, 이 노드는 `region`만을 부모로 하는 CPT로 추출된다. `lang_korean`은 `lifex_languages_lifetime`과 잠재 노드 하나만을 부모로 가지므로 지역이나 문화와 무관하다. 한국 문화 Native라는 값은 `region = East Asia`를 거쳐서만 다른 속성에 영향을 미친다. 그래서 주 언어, 민족, 종교는 동아시아 일반의 분포를 따르고, 한국어 능력, 문화 틀(`cultural_background`), 한국 음식 선호, 정치 성향, 가구 규모, 가구소득의 분포는 전체 비교 집단과 거의 같다(예: 가구소득 2만 5천 달러 미만은 47.7%, 전체도 47.7%). 성격과 가치의 내부 상관도 약하다. 같은 개념을 측정하는 BFI-2 외향성과 IPIP 외향성 하위 요인 평균의 상관은 0.279(전체 0.283)이고, Schwartz 원형에서 마주 보는 가치 다섯 쌍의 상관은 −0.003\~0.011이다.

### 한 페르소나 안의 모순

`demo_checks.py`의 규칙 43개는 나이, 가족, 일, 역할, 학력 사이의 논리적 모순을 평가한다. 세대와 성장기 연대 규칙은 기준 연도를 2026년으로 설정한다(이 저장소의 기준 연도 2020년과 다르다). 아래 비율은 규칙이 적용되는 집단 안의 비율이다.

| 규칙 | 적용 집단(한국) | 한국 | 전체 |
|---|---|---|---|
| 13세 미만인데 소속 조직이 기업, 공공, 학계, NGO | 11,201 | 81.4% | 82.9% |
| 13세 미만인데 어떤 산업이든 Veteran | 11,201 | 59.0% | 58.7% |
| 성인인데 성장기 연대가 나이상 불가능 | 37,237 | 55.6% | 55.9% |
| 학업 경로가 대학원 학위인데 최종학력 학사 미만 | 6,233 | 96.5% | 96.2% |
| 자녀 없음인데 육아 경험 있음 | 28,977 | 30.5% | 30.8% |
| 5세 미만인데 돌봄 제공자 | 4,355 | 22.2% | 23.2% |
| 세대가 나이와 맞지 않음 | 52,864 | 2.4% | 2.3% |
| 계층과 가구소득이 정반대 | 52,864 | 2.1% | 2.2% |

hard 규칙을 하나 이상 어긴 페르소나는 한국 문화 Native의 76.3%(성인 78.2%), 전체의 76.6%(성인 78.6%)다. soft 규칙까지 넣으면 77.1%와 77.4%다. MatrAIx 자체 품질 보고서를 같은 그래프로 다시 돌리면 10,000명 가운데 hard 0명, strong 1명(0.01%)이 검출된다. MatrAIx 저장소에 들어 있는 같은 시드의 보고서(`persona/synthesis/reports/full_dag_quality_10000.json`)는 soft 16명으로 조금 다른데, 원인은 확인하지 않았다. MatrAIx의 점검 규칙(`persona/synthesis/sampler/audit.py`의 `consistency_issues`, 20개)은 대부분 그래프의 금지 규칙이 이미 막는 조합(미성년의 경력, 자녀, 주 언어와 언어 능력, 시각 장애와 운전 등)을 다시 확인하는 것이어서, 그래프에서 연결되지 않은 속성 쌍(노드 쌍의 99.2%) 사이의 모순은 잡지 못한다. 시드 42로 추출한 10,000명에 같은 43개 규칙을 적용한 결과(`seed42_mychecks.json`)도 비율이 비슷하다(예: 13세 미만 소속 조직 79.8%, 대학원 학위 경로 96.5%). 이 결과를 생성한 스크립트는 남아 있지 않다.

### 공개 자료 점검

공개 Persona 1M의 실제 파일은 점검하지 않았다. 따라서 이 장의 수치는 공식 그래프와 샘플러의 재생성 표본에 대한 결과이며, 공개 파일에 포함된 사람 자료 기반 기록의 품질을 평가한 결과는 아니다.

### 비교 수치의 기준

대학원 학위 경로인데 학사 미만인 비율은 저장된 결과에서 96.45%다. 모순이 하나 이상인 비율을 비교할 때는 hard만 집계한 값과 hard, soft를 합친 값을 구분해야 한다. 또한 65세 이상 비율은 모든 연령을 포함하면 약 10%, 성인만 기준으로 하면 14.3%다.

## 2.3 그대로 사용한 것

다음 구성 요소는 MatrAIx의 내용이나 코드를 수정하지 않고 사용한다. 파일은 모두 사용자의 MatrAIx checkout에서 불러오며 이 저장소에 복사하지 않는다(예외는 2.4절의 `dimensions_kr.json`).

| 구성 요소 | MatrAIx 파일 | kr-persona에서 사용하는 곳 | 사용하는 방식 |
|---|---|---|---|
| 속성 1,290개의 id, 이름, 분류, 값 목록 | `persona/schema/dimensions.json`, `full_dag.json`의 `nodes` | `matraix/graph.py`(`graph_values`), `generate/assign.py`(`build`), `generate/masks.py`(`nondev_plan`) | 값 목록을 수정하지 않는다. MatrAIx 범주가 없는 한국 값은 출력에서만 범주 밖 값으로 기록한다(2.4절) |
| 그래프 수치: 기본 확률, 쌍별 연결과 가중치, CPT, 위상 순서 | `persona/synthesis/graph/full_dag.json` | `matraix/sampler.py`(`KRSampler`), `blocks/core.py`(`MX`) | 다시 추정하지 않는다. MatrAIx 속성 1,069개와 잠재 노드 18개는 이 수치로 추출한다 |
| 그래프를 표로 바꾸는 코드: 연결, CPT, 마스크 표, $\gamma_i$, CPT가 대신하는 연결 | `persona/synthesis/sampler/sampler.py`의 `PersonaForwardSampler`(`_compile_pairwise_edges`, `_compile_full_cpts`, `_compile_masks`, `_compile_node_shrinkage`) | `KRSampler.__init__` | `PersonaForwardSampler(graph, SamplingConfig(seed, emit_only=False))`를 생성하여 `in_edges`, `full_cpts`, `masks`, `gamma`, `replaced_parents`, `prior`, `topological_order`를 참조하여 사용한다. 추출 반복문(`sample_indices`)은 사용하지 않는다 |
| 한국 속성이 아닌 노드를 대상으로 하는 MatrAIx 마스크 411개 | `full_dag.json`의 `conditional_masks` | `KRSampler._logits` | 생성 중에 MatrAIx와 같은 방식으로 곱한다. 조건에는 한국 값이 들어간다 |
| MatrAIx 기본 확률에서 정한 구간 경계 | 노드의 `prior` | `generate/base.py`(`TAU`), `blocks/values.py`, `generate/generator.py`(`DERIVED`), `generate/run.py` | 성격 5단계 경계 $z = \pm 0.51, \pm 1.51$(6.5%, 24%, 39%, 24%, 6.5%), 가치 문항과 파생 속성의 경계, 우세 성격 Balanced 26%(경계 $\delta$의 보정 목표)를 MatrAIx 기본 확률에서 가져온다(6장, 7장) |
| 그래프 연결과 CPT를 블록 모형의 사전 평균으로 | `full_dag.json` | `blocks/core.py`(`MX.prior_mean`) | 조사 자료 벌점 로지스틱 모형의 벌점 중심을 MatrAIx 연결과 CPT에서 계산한다(6장) |
| 공개 자료 형식과 변환기 | `persona/post_process/unified_dataset/schema.py`(`UNIFIED_SCHEMA`, `AttributeCodec`) | `generate/run.py` | 그대로 import한다 |
| 공개 전 품질 거르기 규칙 22개 | `persona/post_process/quality_filter/contradictions.json` | `baseline/gen.py` | 규칙 파일을 그대로 불러온다. 적용 코드는 다시 썼다(같은 판정) |
| 샘플러 추출과 품질 보고서 | `PersonaForwardSampler.sample_indices`, `persona/synthesis/scripts/generate_quality_report.py` | `baseline/gen.py`, `artifacts/baseline/qr/` | 2.2절의 재현에만 사용한다 |
| 프롬프트 틀과 라벨 덧씌우기 | `environment/agents/matraix/agents/persona/templates/persona_system.md.j2`, `persona_dimension_catalog.overlay_labels_from_persona_path` | `render/render.py`(`overlay_labels_from_persona_path`가 MatrAIx 함수를 부른다) | 그대로 |

## 2.4 변환한 것

| 구성 요소 | MatrAIx 파일 | kr-persona 파일 | 변환한 내용 |
|---|---|---|---|
| 샘플러 | `persona/synthesis/sampler/sampler.py` | `krpersona/matraix/sampler.py`(`KRSampler`) | 한국 값을 행마다 먼저 고정하고 나머지만 MatrAIx 식으로 추출한다(아래, 7.2절) |
| 한국 속성이 대상인 MatrAIx 마스크 | `full_dag.json`의 `conditional_masks` | `matraix/classify_masks.py`, `matraix/own_masks.py`, `generate/run.py` | 생성 중 금지 규칙에서 제외하고, 일부는 생성 뒤 거부 검사로 옮긴다 |
| 출력 범주 | `dimensions.json`의 값 목록 | `mapping/변수대응표.xlsx`(`범주 대응`), `generate/generator.py` | 스키마는 유지하고 MatrAIx 범주가 없는 한국 값을 출력 값으로 사용한다(4.3절) |
| 속성 목록 파일 | `persona/schema/dimensions.json` | `krpersona/render/dimensions_kr.json` | 1,290개는 순서와 내용이 같고 `cog_assertiveness`의 이름 하나만 바꿨다. 새 한국 속성 21개를 뒤에 덧붙였다 |
| 렌더러 | `src/matraix/persona_dimension_catalog.py` | `krpersona/render/render.py` | 같은 함수 이름과 출력 형식으로 다시 썼다. 값을 제외하지 않는다 |
| 프롬프트 생성 | `environment/agents/matraix/agents/persona/templating.py` | `krpersona/render/templating.patch` | import 한 줄을 `krpersona.render`로 변경한다 |

### 샘플러

`KRSampler`는 MatrAIx의 식을 그대로 다시 구현하되, 한국 속성 집합 $K$(221개, 4.2절)의 값을 생성기가 행마다 정한 뒤 나머지를 추출한다. MatrAIx와 다른 점은 다음과 같으며, 식과 처리 순서는 7.2절에 있다.

- $K$의 값은 MatrAIx 범주 코드로 변환하여 각 행에 입력한다. MatrAIx 범주가 없는 한국 값(예: 주 언어 Korean, 무급가족종사자)은 코드 $|V_a|$("범주 없음")로 포함하고, 이 값에서 나가는 연결과 CPT 항은 인수 1(로그 0)로 설정한다.
- $K$ 밖의 노드(출력 속성 1,069개, 잠재 노드 18개)는 MatrAIx 위상 순서대로 2.1절의 식으로 추출한다. 한국 값은 부모 값으로 들어간다.
- $K$ 밖의 노드 $b$에서 $K$의 노드 $a$로 가는 연결 373개는 방향을 뒤집는다. $a$의 값 $x_a$는 이미 정해져 있으므로, $b$를 추출할 때 다음 항을 더한다.

$$
\log q_b(v) \mathrel{+}= \gamma_b\, w_e \big(\log P_e(x_a \mid v) - \log P^0_a(x_a)\big)
$$

- $K$ 밖 노드의 $\gamma_b$는 새로 받는 연결(뒤집은 연결 포함)의 가중치로 다시 계산한다. 이로써 $\gamma$가 바뀌는 노드는 58개다. $K$의 노드는 MatrAIx의 $\gamma$를 그대로 사용한다.
- $K$ 안의 연결 467개, $K$의 노드가 자식인 CPT 21개, $K$의 노드가 대상인 마스크는 한국 값을 정할 때 사용하지 않는다. 다만 생성기가 한국 범주 하나를 MatrAIx 범주 여러 개 가운데 하나로 나눌 때(`draw_korean`, 예: 미혼 → {Single, In a relationship, Engaged}) 그 허용 집합 안에서 MatrAIx 확률로 고르는 데 사용된다. 이때 아직 정하지 않은 CPT 부모는 기본 확률로 평균한다. 마스크가 허용 집합을 모두 지우면 허용 집합이 우선한다.
- 조건 속성이 아직 추출되지 않은 마스크는 건너뛴다(MatrAIx는 이 경우 오류를 발생시킨다). 마스크를 곱해 모든 값의 확률이 0이 된 행은 그 마스크를 곱하기 전의 확률로 복원한다(MatrAIx는 균등 분포로 복원한다).
- MatrAIx 형식의 추가 마스크(`extra_masks`)와 이미 추출한 값에서 행마다 계산하는 마스크(`dyn_masks`)를 MatrAIx 마스크와 같은 방식으로 곱한다(2.5절).

$K$를 비우면 `KRSampler`는 MatrAIx 샘플러와 동일한 분포를 생성한다. `tests/test_sampler.py`가 이를 속성별 총변동거리로 확인한다. 2026-10-30에 저장한 결과는 5만 명씩 추출했을 때 TVD 중앙값 0.0038, 최댓값 0.0114로, MatrAIx끼리 두 번 추출한 기준선(중앙값 0.0039, 최댓값 0.0129)과 같은 수준이다. MatrAIx 속성의 주변 분포를 한국 집계에 맞추는 추가 보정항 $\alpha$는 적용하지 않았다.

### MatrAIx 마스크

$K$ 밖이 대상인 마스크 411개는 생성 중에 그대로 사용한다. 0단계 생성기의 한국 속성 151개를 대상으로 하는 마스크는 105개이며, 그중 금지 값이 있는 103개를 `matraix/classify_masks.py`로 나눴다. 0단계 생성기로 생성한 6만 명(시드 11, 센서스 기본 집단 `syn_s1`)에서 한국 값이 한 번이라도 어긴 마스크는 한국 값과 충돌한다고 보고 버리고(19개: 센서스 속성 대상 8개, 규칙 속성 대상 11개), 한 번도 어기지 않은 마스크는 생성 뒤 거부 검사로 옮겼다(84개: 센서스 27개, 규칙 46개, 조사 추정 11개). 결과는 `artifacts/masks/mask_class.pkl`과 `docs/tables/matraix_masks.csv`에 있다. 옮긴 84개는 `matraix/own_masks.py`의 `rules`에서 고정 규칙과 함께 생성된 페르소나 전체에 적용되며, 하나라도 어긴 페르소나는 버리고 다시 생성한다(7장).

이 분류에는 두 가지 한계가 있다. 첫째, 0단계 생성기는 조사 추정 속성을 MatrAIx 확률로 임시로 추출했으므로, 조사 추정 속성이 대상인 11개는 최종 한국 모형의 값이 아니라 MatrAIx 값으로 판정되었다. 둘째, 5단계에서 $K$에 더한 속성이 대상인 MatrAIx 마스크 8개(`lang_english` 4개, `dev_professional_status` 3개, `english_proficiency` 1개)는 분류하지 않았고, 대상 값이 한국 모형으로 고정되므로 생성 중에도 생성 뒤에도 사용되지 않는다. 버린 19개도 `draw_korean`으로 허용 집합 안에서 값을 고를 때는 다른 MatrAIx 마스크와 함께 곱해진다.

### 출력 범주

MatrAIx의 값 목록은 변경하지 않는다. 한국 값에 맞는 MatrAIx 범주가 없으면 출력에는 직접 정의한 범주를 사용하고 그래프에는 가장 가까운 MatrAIx 범주를 조건으로 포함한다. 직접 정의 범주 26개와 그 밖의 범주 밖 출력 값(주 언어 Korean, Not applicable, Not a developer 등)은 4.3절에 있다. `dimensions_kr.json`의 값 목록에도 이 값들은 들어 있지 않으므로, 이 파일은 출력 값의 완전한 사전이 아니다. 출력 값의 전체 목록은 `docs/tables/attributes.csv`의 `값 목록` 열과 `category_mapping.csv`에 있다.

### 렌더러

`render/render.py`는 MatrAIx 렌더러와 같은 함수(`collect_dimension_items`, `build_dimension_narrative`, `build_template_context_extras`, `overlay_labels_from_persona_path`)와 같은 출력 형식("### 절", "- 이름: 값")을 사용하며, 다음이 다르다.

1. 값을 제외하지 않는다. MatrAIx 렌더러는 기본값과 같은 값과 "none" 류 값을 빼는데, 한국 페르소나에서 이렇게 제외되는 속성은 페르소나당 평균 670.7개(1,311개 중 51.2%, 최종 표본 `out_on`의 처음 5,000명, `validate/render_check.py`)다. 성격의 Average, 개발 관련 속성의 Not applicable도 정보이므로 모두 기록한다.
2. 직접 정의 범주를 포함해 값을 그대로 기록한다.
3. 빈 값(null)은 생성 오류로 보고 `RenderError`를 발생시킨다.
4. 절과 순서는 MatrAIx와 같고, 새 분류 `Health: Mental`(새 한국 속성 4개)은 Lifestyle & health에 포함한다.
5. Personality & values 절의 첫 줄에 5단계가 한국 성인 안의 상대 위치라는 설명(`LEVEL_GUIDE`)을 포함한다.
6. `dimensions_kr.json`에서 `cog_assertiveness`(Linguistic: Communication)의 이름을 "Assertiveness (communication style)"로 바꿨다. MatrAIx에서는 `big5_assertiveness`와 이름이 같다.

`templating.patch`는 MatrAIx checkout의 `templating.py`에서 `from matraix.persona_dimension_catalog import (...)`를 `from krpersona.render import (...)`로 바꾸는 한 줄짜리 패치다. MatrAIx checkout에서 `git apply`로 적용하며(커밋 `3633d8d`에서 `git apply --check` 통과 확인), `krpersona`가 Python 경로에 있어야 한다.

## 2.5 새로 생성한 것

| 구성 요소 | kr-persona 파일 | MatrAIx에서 이것이 대신하는 것 | 문서 |
|---|---|---|---|
| 한국 속성 집합 $K$(221개)와 생성 방법 배정 | `generate/assign.py`(`build`), `generate/generator.py`(`KR_ADDED`) | 해당 속성의 그래프 추출 | 4.2절 |
| 새 한국 속성 21개 | `blocks/kgss.py`, `blocks/health.py`, `render/dimensions_kr.json` | 없음 | 4.4절 |
| 센서스 기본 집단(순차 CART)과 페르소나 추출 | `census/*`, `generate/base.py`(`Base`), `generate/generator.py`(`Base4`) | 뿌리 노드 86개의 독립 추출과 그 아래 인구 속성 | 5장 |
| 조사 자료 블록 모형(성격, 가치, 건강, 노동, 소득, KGSS, 가족, 개발자 여부) | `blocks/*.py` | 해당 속성의 그래프 추출 | 6장 |
| 규칙 속성과 범주 대응 | `generate/base.py`(`Gen0.make`), `generate/generator.py`(`Gen4.make`) | 해당 속성의 그래프 추출 | 4장, 7장 |
| 성격 점수에서 파생한 속성 37개 | `generate/generator.py`(`DERIVED`, `Gen4._derived`) | MatrAIx 성격 및 선호 속성 37개의 그래프 추출 | `docs/tables/derived_attributes.csv`, 7장 |
| 생성 중 금지 규칙 | `generate/base.py`(`Gen0._extra_masks`, 141개), `generate/masks.py`(`conflict_masks` 286개, `dyn_language_masks` 49개, `nondev_masks` 83개, `rule_pair_masks`), `blocks/kgss.py`(`KPAIRS`) | 없음 | `masks_static.csv`, `masks_compiled.csv`, `nondev_attributes.csv`, 7장, 8장 |
| 생성 뒤 거부 검사와 칸별 재가중 | `matraix/own_masks.py`(`rules`), `generate/run.py` | 없음(MatrAIx에는 생성 중 마스크만 있다) | 5.6절, 7장 |
| 일관성 검사 규칙 1\~4차, 위반 시 MatrAIx 부분 재추출, 규칙의 쌍별 컴파일 | `rules/set1.py`\~`set4.py`, `generate/repair.py`(`RepairRules`), `generate/compile_masks.py` | MatrAIx `consistency_issues` 20개 | 8장 |
| 실제 자료 검증 | `validate/*` | MatrAIx 품질 보고서 | 9장 |
| MatrAIx 재현과 진단 | `baseline/*` | 없음 | 2.2절 |
| 출력 파일 | `generate/run.py` | `persona_codes`, Persona 1M parquet | 아래 |

규칙을 컴파일한 생성 중 마스크(`artifacts/masks/rule_masks.pkl`, 표 `masks_compiled.csv`)는 929개다(금지 칸 2,155개; 8장).

### 출력 형식

`generate/run.py`는 `artifacts/samples/<표본 이름>/`에 다음 파일을 저장한다.

| 파일 | 내용 |
|---|---|
| `personas.parquet` | `persona_id`(`krp-<시드>-<일곱 자리 번호>`)와 출력 속성 1,311개(문자열 범주). MatrAIx 속성을 생성하지 않는 실행(`off`)에서는 한국 속성 242개만 있다 |
| `internal.parquet` | 정확한 나이, 시도, 가구원 수, 소득 금액, 잠재 점수 등 내부 값. 공개하지 않는다 |
| `personas_matraix.parquet` | MatrAIx `UNIFIED_SCHEMA` 형식 |
| `persona_codes.schema.json` | `AttributeCodec.from_codes_schema`가 참조하는 열 목록(1,290개 속성과 MatrAIx 값 목록) |
| `stats.json` | 시도 수, 통과율, 위반 수, 재추출 기록, 상수 |

`personas_matraix.parquet`은 MatrAIx의 `AttributeCodec`으로 생성한다. `source`는 `kr_persona`, `source_record_id`는 `persona_id`이고, `attributes`에는 MatrAIx 속성 1,290개의 4비트 코드가 들어간다. MatrAIx 값 목록에 없는 값은 `attribute_overrides`에 문자열로, 값이 없는 속성은 `null_bitmap`에 적힌다. `has_description`은 거짓이고 `descriptions`, `grounding`, `metadata_json`은 비어 있다. `AttributeCodec`은 열이 정확히 1,290개라고 가정하므로 새 한국 속성 21개는 이 파일에 없고 `personas.parquet`에만 있다. `off` 실행에서는 MatrAIx 속성 1,069개가 모두 null이다. 이 파일은 MatrAIx의 공개 자료 도구(예: `AttributeCodec.decode_row`)로 디코딩할 수 있고, 한국 범주는 `attribute_overrides`로 전달된다.

## 2.6 라이선스와 인용

MatrAIx는 MIT 라이선스(`LICENSE`, "Copyright (c) 2026 MatrAIx")로 배포된다. 이 저장소도 MIT 라이선스("Copyright (c) 2026 kr-persona contributors")다. MIT 라이선스는 소프트웨어의 사본이나 상당 부분을 배포할 때 저작권 표시와 허가 문구를 함께 넣도록 요구한다.

이 저장소에 포함된 MatrAIx 유래 코드와 자료는 다음과 같다. `krpersona/render/dimensions_kr.json`은 MatrAIx `persona/schema/dimensions.json`(속성 정의, `reference_sources`, `personaYamlProbeFields` 포함)을 고친 사본이고, `krpersona/render/templating.patch`에는 MatrAIx `templating.py`의 몇 줄이 들어 있다. 그래프, 샘플러, 변환기, 프롬프트 틀은 사용자의 MatrAIx checkout에서 불러오므로 이 저장소에 없다. 그 밖에 규칙 정의에 사용한 속성 목록(`krpersona/rules/catalogs/*.json`)과 문서 표 `docs/tables/attributes.csv`에도 MatrAIx 속성의 이름, 설명, 값 목록이 들어 있다. 따라서 이 저장소를 배포할 때는 MatrAIx의 저작권 표시와 허가 문구를 함께 넣어야 하며, 그 전문은 `NOTICE.md`에 있다. `dimensions.json`의 `reference_sources`에는 MatrAIx가 참고한 외부 척도와 자료의 이용 조건(예: HEXACO-PI-R은 연구용 참고)이 적혀 있으며, 이 내용은 사본에도 그대로 남아 있다. MatrAIx 그래프의 일부 CPT는 외부 원자료(ACS PUMS 2024, 세계가치관조사 7차, Stack Overflow 2025 조사)로 보정되었고, 그 이용 조건은 MatrAIx 쪽에 있다.

한국 원자료와 그로부터 생성한 모형, 생성한 기본 집단은 각 자료의 이용 조건 때문에 배포하지 않는다(3장).

이 저장소의 결과를 사용하는 연구는 MatrAIx도 인용해야 한다. 출력 속성 1,311개 가운데 1,069개는 MatrAIx 그래프에서 추출하고, 나머지도 MatrAIx의 속성 정의와 범주를 따르기 때문이다. 인용 형식은 2.1절의 BibTeX이다.
