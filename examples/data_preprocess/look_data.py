# 示例：gsm8k/train.parquet 的结构
import pandas as pd
import pyarrow.parquet as pq

# 读取数据
df = pq.read_table('train.parquet').to_pandas()
print(df.columns)  # 应包含 'data_source' 或 'prompt' 字段
print(df.head())