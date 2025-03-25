# Error
# import package_1

"""
try:
    code
except:
    code
else:
    若沒發生錯誤，則執行此區程式碼
finally:
    不管有沒有錯誤都會執行此區程式碼
"""
# raise ValueError('GG') # 手動拋出錯誤
# raise # 單用raise會直接跳到except

# assert #（斷言）用於確保某個條件為 True，否則拋出 AssertionError。
         # 它主要用來在開發和測試階段檢查程式是否符合預期。

import pandas as pd # 資料科學

data = [10, 20, 30, 40]
s = pd.Series(data)
print(s)



