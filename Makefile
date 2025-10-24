.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

INPUT_DATA_FILENAME = "ALL_depth_map_data_202502.csv"

drop_duplicates: ## 重複する座標データを削除して新しいCSVファイルを作成する
	@python3 bin/drop_duplicates.py --input $(INPUT_DATA_FILENAME)


clean: ## データファイルを削除する
	@rm -f data/data_dedup.csv
