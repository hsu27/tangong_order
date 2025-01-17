
# 引入套件
from fastapi import FastAPI,Body, routing, WebSocket, Request, BackgroundTasks   # fastapi相關的模組    # pip install fastapi
from fastapi.middleware.cors import CORSMiddleware  # fastapi相關的模組
import uvicorn  # uvicorn
import threading
from utils import log

ip = '127.0.0.1'     # server ip
port = '3000' # server port


# FastAPI框架
app = FastAPI()

# 建立log
logger=log.build_logging('./log.log')

######################### xxx_sys ##################################

from xxx_sys import main as xxx_sys_main

# xxx_sys-兩數相加
@app.get("/xxx_sys/add_two_numbers/{num1}/{num2}",summary='兩數相加',description='兩數相加',tags=['xxx_sys'])
async def xxx_sys_add_two_numbers(num1: float, num2: float):
    '''
    功能說明:
        兩個數值相加起來，就這樣....
    input:
        num1: 第一個值
        num2: 第二個值
    output:
        result: 執行結果
        sum: 相加結果
    '''
    try:
        data = await routing.run_in_threadpool(xxx_sys_main.xxx_sys_add_two_numbers, num1, num2)
    except Exception as e:
        logger.error(f'{e}')
        logger.error(f'xxx_sys - add_two_numbers')
        data = {'result':False}

    return data

# xxx_sys-兩數相乘
@app.post("/xxx_sys/multiply_two_numbers",summary='兩數相乘',description='兩數相乘',tags=['xxx_sys'])
async def xxx_sys_multiply_two_numbers(num1:float=Body(...,embed=True),num2:float=Body(...,embed=True)):
    '''
    功能說明:
        兩個數值相乘起來，就這樣....
    input:
        num1: 第一個值
        num2: 第二個值
    output:
        result: 執行結果
        multiply: 相乘結果
    '''
    try:
        data = await routing.run_in_threadpool(xxx_sys_main.xxx_sys_multiply_two_numbers, num1, num2)
    except Exception as e:
        logger.error(f'{e}')
        logger.error(f'xxx_sys - multiply_two_numbers')
        data = {'result':False}

    return data

def fun():
    # 撈資料(每月一號執行、計時器)
    # 預測
    # 上傳
    print('測試')


if __name__ == "__main__":
    # Thread
    # a=threading.Thread(target=fun())
    # a.start()


    # FastAPI框架
    uvicorn.run(app, host=ip, port=int(port), log_config="uvicorn_config.json")
