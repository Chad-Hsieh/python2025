import multiprocessing as mp

def check_lottery(ticket_number):
    # 假設中獎號碼為 "12345464"
    winning_number = "12345464"
    result = "有" if ticket_number == winning_number else "無"
    return (ticket_number, result)

if __name__ == "__main__":
    tickets = ["12345464", "XXXXXXXX"]  # 票號列表
    
    with mp.Pool(processes=len(tickets)) as pool:
        results = pool.map(check_lottery, tickets)
    
    # 格式化輸出表格
    print("+----+-------------+--------+")
    print("| #  | 發票號碼    | 中獎情況 |")
    print("+----+-------------+--------+")
    for idx, (ticket, status) in enumerate(results, start=1):
        print(f"| {idx:<2} | {ticket:<11} | {status:<4} |")
    print("+----+-------------+--------+")

"""
mp.Process 手動管理多進程
mp.Pool 自動管理多進程池

進程數量較少（可手動管理） ➝ mp.Process
進程數量較多（需要自動管理） ➝ mp.Pool

如果你的程式要跑 多個不同的函數，用 mp.Process
如果你的程式要跑 同一個函數處理大量數據，用 mp.Pool
"""