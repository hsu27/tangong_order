import pandas as pd
import numpy as np

# 模型平均誤差、尾三月預測值及誤差、最佳模型名稱
def log_create():
    # Define the correct column names
    correct_columns = ['模型名稱', '預測月份', '預測值', '誤差值', 'MAE']
    # Initialize the DataFrame with empty content
    log_df = pd.DataFrame(columns=correct_columns)
    
    return log_df

def log_append(log_df, model_name, date, predict_value, true_value, mae):
    new_row = {
        '模型名稱': model_name,
        '預測月份': date,
        '預測值': predict_value,
        '誤差值': true_value,
        'MAE': mae
    }
    log_df.loc[len(log_df)] = new_row

def log_save(log_df):

    # Save the processed data to a new CSV file
    output_file_path = '/mnt/data/processed_output_log.csv'
    final_data.to_csv(output_file_path, index=False, encoding='utf-8-sig')