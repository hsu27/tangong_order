import pandas as pd
import numpy as np
import os
import datetime

# 模型平均誤差、尾三月預測值及誤差、最佳模型名稱
def log_create():
    # Define the correct column names
    correct_columns = ['模型名稱', '預測月份', '預測值', '誤差值', 'MAE']
    # Initialize the DataFrame with empty content
    log_df = pd.DataFrame(columns=correct_columns)
    
    return log_df

def log_append(log_df, model_name, date, predict_value, true_value, mae):
    # Calculate the absolute error
    error_value = abs(predict_value - true_value)
    print(true_value)

    # Add a new row
    new_row = {
        '模型名稱': model_name,
        '預測月份': date,
        '預測值': predict_value,
        '誤差值': error_value,
        'MAE': mae
    }
    log_df.loc[len(log_df)] = new_row

    return log_df

def log_save(log_df):
    # Get the current date in YYYYMMDD format
    current_date = datetime.datetime.now().strftime('%Y%m%d')

    # Generate the file name
    file_name = f"{current_date}_log.csv"

    # Define the full file path
    output_file_path = os.path.join('./log_data', file_name)

    # Save the DataFrame to a CSV file
    log_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    # Return the file path
    return output_file_path