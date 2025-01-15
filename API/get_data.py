import requests
import json
import os
import pandas as pd

def get_item_attributes(item_code, data):
    for item in data.get("data", []):
        if item.get("item_code") == item_code:
            return {
                "type": item.get("type"),
                "mg": item.get("mg"),
                "material": item.get("material"),
                "sp_size": item.get("sp_size"),
                "sp_size2": item.get("sp_size2")
            }
    return None

def sanitize_filename(filename):
    # 移除或替換無效字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def adjust_sp_size(material, sp_size, item_data):
    # 找尋最接近目前 sp_size 的尺寸
    candidates = [
        float(item.get("sp_size"))
        for item in item_data.get("data", [])
        if item.get("material") == material
    ]
    if not candidates:
        return sp_size
    return min(candidates, key=lambda x: abs(x - float(sp_size)))

def load_or_fetch_json(url, filename):
    # 檢查檔案是否已經存在
    if os.path.exists(filename):
        # 如果檔案存在，從本地載入資料
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        # 如果檔案不存在，從 API 獲取資料並儲存到本地端
        response = requests.get(url)
        response.raise_for_status()  # 確保請求成功
        data = response.json()  # 將回應轉換為 JSON 格式
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)  # 儲存到本地檔案
        return data

if __name__ == '__main__':
    output_folder = './data'
    os.makedirs(output_folder, exist_ok=True)

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

    # 設定 JSON 檔案名稱
    all_data_file = "all_data.json"
    item_data_file = "item_data.json"

    try:
        # 讀取或取得 JSON 資料
        all_data = load_or_fetch_json(all_url, all_data_file)
        item_data = load_or_fetch_json(item_info_url, item_data_file)

        # 儲存統一格式的資料
        unified_details = []

        for item in all_data.get("data", []):
            item_code = str(item["item_code"]).upper()
            cus_abbr = item.get("cus_abbr", "Unknown")
            validiay = item.get("validiay", "Unknown")
            weight = float(item.get("weight", 0))

            if weight == 0:
                continue

            attributes = get_item_attributes(item_code, item_data)
            
            if attributes:
                item_type = attributes.get("type")
                material = attributes.get("material")
                mg = attributes.get("mg")
                sp_size = attributes.get("sp_size")
                sp_size2 = attributes.get("sp_size2", 0)

                if mg is None:
                    mg = material_mapping.get(material)
            else:
                # 舊編碼，API 無對應資料
                # CH00G4000D1.5875
                # 
                g_index = item_code.find("G")
                if g_index == -1:
                    continue

                r_index = item_code.find("R", g_index + 1)
                d_index = item_code.find("D", g_index + 1)

                if r_index != -1:
                    item_type = "R"
                    material = item_code[:g_index]
                    parts = item_code[r_index + 1:].split("X")
                    sp_size = parts[0]
                    sp_size2 = parts[1] if len(parts) > 1 else 0
                elif d_index != -1:
                    item_type = "B"
                    material = item_code[:g_index]
                    sp_size = item_code[d_index + 1:]
                    sp_size2 = 0
                else:
                    continue

                mg = material_mapping.get(material)
                if mg is None:
                    continue

            sp_size = adjust_sp_size(material, sp_size, item_data)

            unified_details.append({
                "item_code": item_code,
                "item_type": item_type,
                "material": material,
                "mg": mg,
                "sp_size": sp_size,
                "sp_size2": sp_size2,
                "cus_abbr": cus_abbr,
                "validiay": validiay,
                "weight": weight
            })

        # 將 unified_details 儲存為 JSON 檔案
        with open("unified_details.json", 'w') as json_file:
            json.dump(unified_details, json_file, indent=4)
        print(f"已儲存 unified_details")

        # 將統一格式的資料直接轉換為 DataFrame
        df = pd.DataFrame(unified_details)

        # 將有效日期轉換為日期時間格式
        df["validiay"] = pd.to_datetime(df["validiay"], errors="coerce")
        df["year_month"] = df["validiay"].dt.to_period("M")
            
        # 根據指定的列進行分組
        groups = df.groupby(["cus_abbr", "mg", "sp_size"])            

        # 處理每個分組
        for group_keys, group_data in groups:
            # 根據分組的鍵生成文件名
            filename = "_".join(map(str, group_keys)) + ".csv"
            filename = sanitize_filename(filename)
            filepath = os.path.join(output_folder, filename)

            # 檢查分組的資料量是否大於 100
            if len(group_data) > 100:
                # 計算 Z-score
                group_data['z_score'] = (group_data['weight'] - group_data['weight'].mean()) / group_data['weight'].std()
                
                # 將異常值 (|Z| > 3) 的 weight 設為 0
                group_data.loc[group_data['z_score'].abs() > 3, 'weight'] = 0
                    
            # 獲取最小和最大日期，確保完整日期範圍
            min_date = group_data["validiay"].min()
            max_date = group_data["validiay"].max()
            
            # 創建完整的日期範圍
            date_range = pd.date_range(
                start=min_date.replace(day=1),
                end=max_date.replace(day=1),
                freq='MS'
            )
            
            # 使用完整的日期範圍創建基礎DataFrame
            result_df = pd.DataFrame({'date': date_range})
            
            # 按年-月匯總重量
            monthly_sums = group_data.groupby(pd.Grouper(key='validiay', freq='MS'))['weight'].sum()
            
            # 將月度總和與完整日期範圍合併
            result_df = result_df.merge(
                monthly_sums.reset_index(),
                left_on='date',
                right_on='validiay',
                how='left'
            )
            
            # 用0填充缺失值
            result_df['order'] = result_df['weight'].fillna(0)
            
            # 只保留需要的列
            result_df = result_df[['date', 'order']]
            
            # 將日期格式化為YYYY-MM
            result_df['date'] = result_df['date'].dt.strftime('%Y-%m')
            
            # 儲存為CSV文件
            result_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"已儲存: {filepath}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP 請求錯誤：{e}")
    except KeyError as e:
        print(f"資料處理錯誤：缺少鍵值 {e}")