.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# dataディレクトリにあるCSVファイルを処理する

# 入力ファイル名(data/は含まず、ファイル名のみを指定)
DATA_FILENAME = ALL_depth_map_data_202510.csv
BASENAME = $(basename $(DATA_FILENAME))

DE=_de
DEDUP=_dd
OUTLIER=_ol
INTERPOLATE=_ip
MEDIAN_FILTER=_mf

de: ## step0. drop epoch epoch列を削除して新しいCSVファイルを作成する
	@python3 bin/process_drop_epoch.py --input $(DATA_FILENAME) --output $(BASENAME)$(DE).csv

dd: de ## step1. deduplicate 重複する座標データを削除して新しいCSVファイルを作成する
	@python3 bin/process_duplicate.py --input $(BASENAME)$(DE).csv --output $(BASENAME)$(DE)$(DEDUP).csv

ol: dd ## step2. outlier 外れ値を検出して新しいCSVファイルを作成する
	@python3 bin/process_outlier.py --input $(BASENAME)$(DE)$(DEDUP).csv --output $(BASENAME)$(DE)$(DEDUP)$(OUTLIER).csv

ip: ol ## step3. interpolate 欠損値を補間して新しいCSVファイルを作成する
	@python3 bin/process_interpolate.py --input $(BASENAME)$(DE)$(DEDUP)$(OUTLIER).csv --output $(BASENAME)$(DE)$(DEDUP)$(OUTLIER)$(INTERPOLATE).csv

mf: ip ## step4. median filter メディアンフィルタを適用して新しいCSVファイルを作成する
	@python3 bin/process_median_filter.py --input $(BASENAME)$(DE)$(DEDUP)$(OUTLIER)$(INTERPOLATE).csv --output $(BASENAME)$(DE)$(DEDUP)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv

all: mf ## 重複排除→外れ値除去→補間→メディアンフィルタを順に全て実行して最終データを作成する
	@cp data/$(BASENAME)$(DE)$(DEDUP)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv static/data/bathymetric_data.csv

scatter: ## 散布図を作成する
	@if [ -f data/$(BASENAME)$(DE).csv ]; then \
		python3 bin/draw_scatter.py --input $(BASENAME)$(DE).csv --title "Original"; \
	fi
	@if [ -f data/$(BASENAME)$(DE)$(DEDUP).csv ]; then \
		python3 bin/draw_scatter.py --input $(BASENAME)$(DE)$(DEDUP).csv --title "Deduplicated"; \
	fi
	@if [ -f data/$(BASENAME)$(DE)$(DEDUP)$(OUTLIER).csv ]; then \
		python3 bin/draw_scatter.py --input $(BASENAME)$(DE)$(DEDUP)$(OUTLIER).csv --title "Outlier removed"; \
	fi

geojson: ## 凸包を作成してGeoJSONファイルを出力する
	@python3 bin/create_convex_hull_geojson.py --input $(BASENAME)$(DE)$(DEDUP)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv --output $(BASENAME).geojson


clean: ## 中間データファイルを削除する
	@rm -f data/*_de.csv
	@rm -f data/*_dd.csv
	@rm -f data/*_dd_ol.csv
	@rm -f data/*_dd_ol_ip.csv
	@rm -f data/*_dd_ol_ip_mf.csv
