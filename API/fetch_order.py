import requests
from datetime import datetime, timedelta
import json
import os

# API URL
all_url = "http://192.168.22.20:6001/data_access_layer/select_all_ship_data"
item_info_url = "http://192.168.22.20:6001/data_access_layer/select_all_item_info"

# 材質與材質群組對應代碼的對應表
material_mapping = {
 'CH00': 'CH', 'CH01': 'CH', 'CH02': 'CH',
 'X46CR13': 'S2', '4C13': 'S2', 'X46C': 'S2', 'SS420C': 'S2', '420J': 'S2',
 'SS440C': 'SC', '420A': 'SC', 'SS440': 'SC', '440C': 'SC', 'SS440AHLW': 'SC',
 'BR00': 'BR',
 '304L': 'M04', 'SS304': 'M04', 'S304': 'M04', 'SS304L': 'M04', 'SS304HC': 'M04',
 '316L': 'L16',
 'XM19': 'XM',
 'SI3N4': 'SN',
 'S316': 'S16', 'S316L': 'S16', 'SS316L': 'S16',
 'PP00': 'PP', 'PP': 'PP',
 'PM00': 'P1', 'PM10': 'P1', 'PM23': 'P1', 'POM100': 'P1',
 'PM5B': 'P5', 'PM50': 'P5', 'PM52': 'P5', 'PM90': 'P5', 'POM500': 'P5',
 'NY6R': 'NL', 'NY66': 'NL',
 'C1X': 'C10', 'CA10': 'C10', 'SCA10': 'C10', 'CA10P': 'C10', 'CA10CP': 'C10', 'CA10NP': 'C10',
 'CA06': 'C06', 'CA08': 'C06',
 'CA15': 'C15', 'CA18': 'C15',
 'CA22': 'C22',
}

def generate_months_between_exclusive(date1, date2):
    '''日期字串'''
    # 轉換字串為日期物件
    start_date = datetime.strptime(date1, "%Y-%m")
    end_date = datetime.strptime(date2, "%Y-%m")
    
    # 初始化結果列表
    months_list = []
    
    # 迴圈逐月遞增，直到超過結束日期，排除頭尾
    current_date = start_date + timedelta(days=32)  # 開始時跳過 start_date
    current_date = current_date.replace(day=1)  # 確保回到月初
    while current_date < end_date:
        months_list.append(current_date.strftime("%Y-%m"))
        current_date += timedelta(days=32)
        current_date = current_date.replace(day=1)  # 確保回到月初

    return months_list

def load_or_fetch_json(url, filepath):
    # 檢查檔案是否已經存在
    if os.path.exists(filepath):
        # 如果檔案存在，從本地載入資料
        with open(filepath, 'r') as file:
            return json.load(file)
    else:
        # 如果檔案不存在，從 API 獲取資料並儲存到本地端
        response = requests.get(url)
        response.raise_for_status()  # 確保請求成功
        data = response.json()  # 將回應轉換為 JSON 格式
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)  # 儲存到本地檔案
        
        response.close()
        return data


def extract_data ():
    # 取得當前時間
    current_date = datetime.now()
    # 計算上一個月
    if current_date.month == 1:
        # 如果是 1 月，上一個月是去年的 12 月
        last_month_date = current_date.replace(year=current_date.year - 1, month=12)
    else:
        # 否則，將月份減 1
        last_month_date = current_date.replace(month=current_date.month - 1)
    # 格式化為年-月
    last_month_year_month = last_month_date.strftime("%Y-%m")

    output_folder = './data/'
    os.makedirs(output_folder, exist_ok=True)

    # 訂單資料
    response = requests.get(all_url,timeout=30)
    response = response.json()
    order_data = response['data']

    # 產邊對照表
    response = requests.get(item_info_url,timeout=30)
    response = response.json()
    item_data = response['data']
    item_table = {} # 對照表
    for item_info in item_data:
        item_table[item_info['item_code']] = item_info

    # 資料處理及歸類
    classifyed_data = {}
    for order_info in order_data:
        # 編碼對應表
        if order_info['item_code'] in item_table:
            item_info = item_table[order_info['item_code']]
        else:
            # print(f"{order_info['item_code']}對應不到!!")
            continue

        # 新舊編號處理
        if order_info['item_code'][0] in ['B','R']:
            # 新編碼
            order_info['item_type'] = item_info['type']   # 類型
            order_info['material'] = item_info['material'] # 材質
            order_info['mg'] = item_info['mg']           # 材質群組
            order_info['sp_size'] = item_info['sp_size']   # 尺寸
            order_info['sp_size2'] = item_info['sp_size2'] # 尺寸 2
        else:
            # 舊編碼
            order_info['item_type'] = item_info['type']   # 類型
            # 材質轉碼
            if item_info['material'] in material_mapping:
                mg = material_mapping[item_info['material']]
            else:
                # print(f"{item_info['material']}對應不到!!")
                continue
            order_info['material'] = mg # 材質
            order_info['mg'] = mg           # 材質群組
            order_info['sp_size'] = item_info['sp_size']   # 尺寸
            if item_info['sp_size2'] is None:
                item_info['sp_size2'] = 0
            order_info['sp_size2'] = item_info['sp_size2'] # 尺寸2
        
        # 同產品客戶歸類(客戶名稱-類型-材質群組-尺寸-尺寸2)
        classify_key = f"{order_info['cus_abbr']}-{order_info['item_type']}-{order_info['mg']}-{order_info['sp_size']}-{order_info['sp_size2']}"
        if classify_key not in classifyed_data:
            classifyed_data[classify_key] = {}
            classifyed_data[classify_key]['cus_abbr'] = order_info['cus_abbr']    # 客戶名稱
            classifyed_data[classify_key]['item_type'] = order_info['item_type']    # 類型
            classifyed_data[classify_key]['mg'] = order_info['mg']    # 材質
            classifyed_data[classify_key]['sp_size'] = order_info['sp_size']    # 尺寸
            classifyed_data[classify_key]['sp_size2'] = order_info['sp_size2']    # 尺寸2
            classifyed_data[classify_key]['data'] = {order_info['validiay'][:7]:order_info['weight']}    # 日期:重量
            classifyed_data[classify_key]['date_list'] = [order_info['validiay'][:7]] # 已有的日期清單
        else:
            # 產生空白日期
            date_list = generate_months_between_exclusive(classifyed_data[classify_key]['date_list'][-1],order_info['validiay'][:7])
            if len(date_list) > 0:
                for date in date_list:
                    classifyed_data[classify_key]['data'][date] = 0    # 日期:重量
                    classifyed_data[classify_key]['date_list'].append(date) # 已有的日期清單
            # 加入資料
            if order_info['validiay'][:7] in classifyed_data[classify_key]['date_list']:
                classifyed_data[classify_key]['data'][order_info['validiay'][:7]] +=order_info['weight']
            else:
                classifyed_data[classify_key]['data'][order_info['validiay'][:7]] = order_info['weight']    # 日期:重量
                classifyed_data[classify_key]['date_list'].append(order_info['validiay'][:7]) # 已有的日期清單
    # 格式轉換
    for classify_key, cus_prod_data in classifyed_data.items():
        data_convert_format = []
        for date,weight in cus_prod_data['data'].items():
            data_convert_format.append({'date':date, 'order':weight})
        # 空白時間填空
        date_list = generate_months_between_exclusive(cus_prod_data['date_list'][-1],last_month_year_month)
        for date in date_list:
            data_convert_format.append({'date':date, 'order':0})
        # 覆蓋
        classifyed_data[classify_key]['data'] = data_convert_format

    # print(classifyed_data)
    return classifyed_data