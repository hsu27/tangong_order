import logging

# 建立logging
def build_logging(log_file):
    logger = logging.getLogger(__name__)#建立logging
    logger.setLevel(level=logging.INFO)#設定顯示等級
    handler1 = logging.FileHandler(log_file)#設定第一個輸出handler1 輸出檔案檔名
    handler2 = logging.StreamHandler()#設定第二個輸出handler2 輸出Terminal
    formatter = logging.Formatter('%(asctime)s - %(lineno)s - %(levelname)s - %(message)s')#設定資料格式
    handler1.setFormatter(formatter)#將格式套用在handler1 
    handler2.setFormatter(formatter)#將格式套用在handler2 
    logger.addHandler(handler1)#handler1新增至logger
    logger.addHandler(handler2)#handler2新增至logger

    #logger.debug('debug message')
    #logger.info('info message')
    #logger.warning('warning message')
    #logger.error('error message')
    #logger.critical('critical message')

    return logger

# 執行log紀錄
def run_logging():
    logger = logging.getLogger(__name__)#建立logging


    return logger