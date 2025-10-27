.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

INPUT_DATA_FILENAME = ALL_depth_map_data_202502.csv

DEDUP=_dd
OUTLIER=_ol
INTERPOLATE=_ip

INPUT_DATA_BASENAME = $(basename $(INPUT_DATA_FILENAME))

dedup: ## 重複する座標データを削除して新しいCSVファイルを作成する
	@python3 bin/process_duplicates.py --input $(INPUT_DATA_FILENAME) --output $(INPUT_DATA_BASENAME)$(DEDUP).csv

outliers: ## 外れ値を検出して新しいCSVファイルを作成する
	@python3 bin/process_outliers.py --input $(INPUT_DATA_BASENAME)$(DEDUP).csv --output $(INPUT_DATA_BASENAME)$(DEDUP)$(OUTLIER).csv

interpolate: ## 欠損値を補完して新しいCSVファイルを作成する
	@python3 bin/process_interpolation.py --input $(INPUT_DATA_BASENAME)$(DEDUP)$(OUTLIER).csv --output data/data_interpolated.csv

scatter: ## 散布図を作成する
	@python3 bin/draw_scatter.py --input $(INPUT_DATA_BASENAME)$(DEDUP).csv --title "Deduplicated"
	@python3 bin/draw_scatter.py --input $(INPUT_DATA_BASENAME)$(DEDUP)$(OUTLIER).csv --title "Outlier removed"

clean: ## データファイルを削除する
	@rm -f data/data_dedup.csv
