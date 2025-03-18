import multiprocessing as mp
import time

WINNING_NUMBERS = {"12345464", "99999999"}

def check_invoice(index, invoice_number, result_list):
    # 檢查發票是否中獎、結果加入列表
    time.sleep(1)
    result_list.append((index, invoice_number, "有" if invoice_number in WINNING_NUMBERS else "無"))

def print_table(results):
    print("+----+-------------+--------+")
    print("| #  | 發票號碼    | 中獎情況 |")
    print("+----+-------------+--------+")

    for i, (_, num, status) in enumerate(results, start=1):
        print(f"| {str(i).ljust(2)} | {num.ljust(11)} | {status.ljust(4)} |")

    print("+----+-------------+--------+")

if __name__ == "__main__":
    invoices = ["12345464", "XXXXXXXX", "99999999"]
    start = time.time()

    results = []
    process_list = []
    with mp.Manager() as manager:
        result_list = manager.list()
        for idx, invoice in enumerate(invoices):  # 加上index確保順序
            process = mp.Process(target=check_invoice, args=(idx, invoice, result_list))
            process_list.append(process)
            process.start()

        for process in process_list:
            process.join()

        results = sorted(result_list)  # 確保與input順序一致

    print("Total costing time:", time.time() - start)

    print_table(results)

"""
010 023
004 025
024 我
012 002
"""
