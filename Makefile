.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# dataディレクトリにあるCSVファイルを処理する

# 入力ファイル名(data/は含まず、ファイル名のみを指定)
DATA_FILENAME = ALL_depth_map_data_202510.csv
BASENAME = $(basename $(DATA_FILENAME))

# Prefer venv python if available
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

DE=_de
DEDUP=_dd
DENSITY_REDUCE=_dr
OUTLIER=_ol
INTERPOLATE=_ip
MEDIAN_FILTER=_mf
UNIFORM_RANDOM=_ur

# density reduce params (override via `make dr DR_RADIUS_M=...`)
DR_RADIUS_M ?= 1.0

# uniform random fill params (override via `make ur UR_N=...`)
UR_N ?= 10000
UR_SEED ?= 42
UR_K ?= 8
UR_POWER ?= 2.0
# ランダムに配置した点を考慮する際の最大距離 (meters)
UR_MAX_DIST ?= 50
# 範囲内にある点の数をカウントするための半径 (meters)
UR_DENSE_RADIUS ?= 10
# 範囲内にある点の数が多い場合、すなわちデータ密度が高い場合に、その点を無視するための閾値
UR_DENSE_MAX_NEIGHBORS ?= 200
# ランダムに配置した点のみを出力するかどうか (1: yes, 0: no)
UR_ONLY_GENERATED ?= 0

UR_ARGS = --n $(UR_N) --seed $(UR_SEED) --k $(UR_K) --power $(UR_POWER)
ifneq ($(strip $(UR_MAX_DIST)),)
UR_ARGS += --max-distance-m $(UR_MAX_DIST)
endif
ifneq ($(strip $(UR_DENSE_RADIUS)),)
UR_ARGS += --dense-radius-m $(UR_DENSE_RADIUS)
endif
ifneq ($(strip $(UR_DENSE_MAX_NEIGHBORS)),)
UR_ARGS += --dense-max-neighbors $(UR_DENSE_MAX_NEIGHBORS)
endif
ifeq ($(UR_ONLY_GENERATED),1)
UR_ARGS += --only-generated
endif

de: ## step0. drop epoch epoch列を削除して新しいCSVファイルを作成する
	@$(PYTHON) bin/process_drop_epoch.py --input $(DATA_FILENAME) --output $(BASENAME)$(DE).csv

dd: de ## step1. deduplicate 重複する座標データを削除して新しいCSVファイルを作成する
	@$(PYTHON) bin/process_duplicate.py --input $(BASENAME)$(DE).csv --output $(BASENAME)$(DE)$(DEDUP).csv

dr: dd ## step1.5 density reduce 半径nメートル以内の点群を1点に集約して点数を減らす
	@$(PYTHON) bin/process_density_reduce.py --input $(BASENAME)$(DE)$(DEDUP).csv --output $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE).csv --radius-m $(DR_RADIUS_M)

ol: dr ## step2. outlier 外れ値を検出して新しいCSVファイルを作成する
	@$(PYTHON) bin/process_outlier.py --input $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE).csv --output $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE)$(OUTLIER).csv

#
# ランダムに点を配置して補完する処理
#
ur: ol ## step2.5 uniform random fill 矩形領域に一様ランダム点を置いてIDWで水深を推定し補完する
	@$(PYTHON) bin/process_uniform_random_fill.py --input $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE)$(OUTLIER).csv --output $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE)$(OUTLIER)$(UNIFORM_RANDOM).csv $(UR_ARGS)

#
# 四分木を使って補完する処理
#
ip: ol ## step3. interpolate 四分木で補間して新しいCSVファイルを作成する
	@$(PYTHON) bin/process_interpolate.py --input $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE)$(OUTLIER).csv --output $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE)$(OUTLIER)$(INTERPOLATE).csv

#
# メディアンフィルタ処理
#
mf: ip ## step4. median filter メディアンフィルタを適用して新しいCSVファイルを作成する
	@$(PYTHON) bin/process_median_filter.py --input $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE)$(OUTLIER)$(INTERPOLATE).csv --output $(BASENAME)$(DE)$(DEDUP)$(DENSITY_REDUCE)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv

#
# 現時点でのallはmf
#
all: mf ## 重複排除→外れ値除去→補間→メディアンフィルタを順に全て実行して最終データを作成する
	@cp data/$(BASENAME)$(DE)$(DEDUP)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv static/data/bathymetric_data.csv

scatter: ## 散布図を作成する
	@if [ -f data/$(BASENAME)$(DE).csv ]; then \
		$(PYTHON) bin/draw_scatter.py --input $(BASENAME)$(DE).csv --title "Original"; \
	fi
	@if [ -f data/$(BASENAME)$(DE)$(DEDUP).csv ]; then \
		$(PYTHON) bin/draw_scatter.py --input $(BASENAME)$(DE)$(DEDUP).csv --title "Deduplicated"; \
	fi
	@if [ -f data/$(BASENAME)$(DE)$(DEDUP)$(OUTLIER).csv ]; then \
		$(PYTHON) bin/draw_scatter.py --input $(BASENAME)$(DE)$(DEDUP)$(OUTLIER).csv --title "Outlier removed"; \
	fi

geojson: ## 凸包を作成してGeoJSONファイルを出力する
	@$(PYTHON) bin/create_geojson.py \
		--input $(BASENAME)$(DE)$(DEDUP)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv \
		--output $(BASENAME).geojson \
		--name "Aburatsubo" \
		--description "bathymetric data in Aburatsubo area collected by Deeper Sonar" \
		#--alpha 0.001 \
		#--no-depth-polygons \
		#--no-contours \


clean: ## 中間データファイルを削除する
	@rm -f data/*_de.csv
	@rm -f data/*_dd.csv
	@rm -f data/*_dd_dr.csv
	@rm -f data/*_dd_ol.csv
	@rm -f data/*_dd_dr_ol.csv
	@rm -f data/*_dd_dr_ol_ur.csv
	@rm -f data/*_dd_dr_ol_ip.csv
	@rm -f data/*_dd_dr_ol_ip_mf.csv
