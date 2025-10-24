.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

INPUT_DATA_FILENAME = ALL_depth_map_data_202502.csv
INPUT_DATA_BASENAME = $(basename $(INPUT_DATA_FILENAME))

dedup: ## 重複する座標データを削除して新しいCSVファイルを作成する
	@python3 bin/process_duplicates.py --input $(INPUT_DATA_FILENAME) --output $(INPUT_DATA_BASENAME)_dedup.csv

image: ## 散布図を作成する
	@python3 bin/draw_scatter.py --input $(INPUT_DATA_BASENAME)_dedup.csv --title "Deduplicated Data Scatter Plot"


clean: ## データファイルを削除する
	@rm -f data/data_dedup.csv
