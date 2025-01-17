from utils import log


# xxx_sys-兩數相加
def xxx_sys_add_two_numbers(num1, num2):
    logger = log.run_logging()   # 執行log
    logger.info(f"adding two numbers, {num1} and {num2}")
    return {'reault': True ,'sum':num1+num2}


# xxx_sys-兩數相乘
def xxx_sys_multiply_two_numbers(num1,num2):
    logger = log.run_logging()   # 執行log
    logger.info(f"multiplying two numbers, {num1} and {num2}")
    return {'reault': True,'multiply':num1*num2}