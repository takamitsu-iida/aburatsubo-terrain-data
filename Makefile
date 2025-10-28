.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# dataディレクトリにあるCSVファイルを処理する

# 入力ファイル名(data/は含まず、ファイル名のみを指定)
DATA_FILENAME = ALL_depth_map_data_202510.csv
DATA_BASENAME = $(basename $(DATA_FILENAME))

DEDUP=_dd
OUTLIER=_ol
INTERPOLATE=_ip
MEDIAN_FILTER=_mf

dd: ## step1. deduplicate 重複する座標データを削除して新しいCSVファイルを作成する
	@python3 bin/process_duplicate.py --input $(DATA_FILENAME) --output $(DATA_BASENAME)$(DEDUP).csv

ol: dd ## step2. outlier 外れ値を検出して新しいCSVファイルを作成する
	@python3 bin/process_outlier.py --input $(DATA_BASENAME)$(DEDUP).csv --output $(DATA_BASENAME)$(DEDUP)$(OUTLIER).csv

ip: dd ol ## step3. interpolate 欠損値を補間して新しいCSVファイルを作成する
	@python3 bin/process_interpolate.py --input $(DATA_BASENAME)$(DEDUP)$(OUTLIER).csv --output $(DATA_BASENAME)$(DEDUP)$(OUTLIER)$(INTERPOLATE).csv

mf: dd ol ip ## step4. median filter メディアンフィルタを適用して新しいCSVファイルを作成する
	@python3 bin/process_median_filter.py --input $(DATA_BASENAME)$(DEDUP)$(OUTLIER)$(INTERPOLATE).csv --output $(DATA_BASENAME)$(DEDUP)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv

all: dd ol ip mf ## 重複排除→外れ値除去→補間→メディアンフィルタを順に全て実行
	@cp data/$(DATA_BASENAME)$(DEDUP)$(OUTLIER)$(INTERPOLATE)$(MEDIAN_FILTER).csv static/data/processed_data.csv

scatter: ## 散布図を作成する
	@if [ -f data/$(DATA_BASENAME)$(DEDUP).csv ]; then \
		python3 bin/draw_scatter.py --input $(DATA_BASENAME)$(DEDUP).csv --title "Deduplicated"; \
	fi
	@if [ -f data/$(DATA_BASENAME)$(DEDUP)$(OUTLIER).csv ]; then \
		python3 bin/draw_scatter.py --input $(DATA_BASENAME)$(DEDUP)$(OUTLIER).csv --title "Outlier removed"; \
	fi

clean: ## データファイルを削除する
	@rm -f data/data_dedup.csv
