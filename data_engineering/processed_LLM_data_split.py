#%%
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile('../data/announcement/processed_LLM_data_3_10_30_2k_summary.parquet')
LLM_data = parquet_file.read().to_pandas()

train_test_split = "2023-04-14 09:00:00"
LLM_data_train = LLM_data[LLM_data.loc[:, "time"] < train_test_split]
LLM_data_test = LLM_data[LLM_data.loc[:, "time"] >= train_test_split]

table = pa.Table.from_pandas(LLM_data_train)
pq.write_table(table, '../data/announcement/processed_LLM_data_train_3_10_30_2k_summary.parquet')
table = pa.Table.from_pandas(LLM_data_test)
pq.write_table(table, '../data/announcement/processed_LLM_data_test_3_10_30_2k_summary.parquet')
# %%
