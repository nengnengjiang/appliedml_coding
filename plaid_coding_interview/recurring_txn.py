'''
Input:
[
  ("Netflix", 9.99, 0),
  ("Netflix", 9.99, 10),
  ("Netflix", 9.99, 20),
  ("Netflix", 9.99, 30),
  ("Amazon", 27.12, 32),
  ("Sprint", 50.11, 45),
  ("Sprint", 50.11, 55),
  ("Sprint", 50.11, 65),
  ("Sprint", 60.13, 77),
  ("Netflix", 9.99, 50),
]

Part 1
Minimum 3 transactions are required to consider it as a recurring transaction
Same company, same amount, same number of days apart - recurring transactions

Input: Company, Amount, Timestamp (Day of the transaction)
Output: An array of companies who made recurring transactions

'''


''' code for part 1'''

from collections import defaultdict

def find_recurring_companies_strict(transactions):
    """
    Strict version:
    A company is recurring if it has 3 consecutive transactions
    with identical amounts and identical day gaps.
    """
    by_company = defaultdict(list)
    for name, amt, day in transactions:
        by_company[name].append((day, amt))
    res = []
    for name, arr in by_company.items():
        arr.sort()  # sort by day
        if _has_recurring_strict(arr):
            res.append(name)
    return res

def _has_recurring_strict(day_amt_list):
    n = len(day_amt_list)
    if n < 3:
        return False
    for i in range(n - 2):
        d1, a1 = day_amt_list[i]
        d2, a2 = day_amt_list[i + 1]
        d3, a3 = day_amt_list[i + 2]
        # strict amount match
        if a1 != a2 or a2 != a3:
            continue
        gap1 = d2 - d1
        gap2 = d3 - d2
        # strict gap match
        if gap1 == gap2 and gap1 > 0:
            return True
    return False

'''
Part 2
The amounts and timestamps - can be similar within a tolerant range
the range is defined by (min of value, 120% of min value)
this applies to both amounts and time delta (day gaps)
'''

''' code for part 2 - allow a range rather than 
strictly same value for amount and time delta '''

from collections import defaultdict

def find_recurring_companies_tolerant(transactions, pct=0.20):
    """
    Follow-up tolerant version:
    Same structure as strict, but window matches if:
      - amounts within [min_amt, min_amt*(1+pct)]
      - gaps within   [min_gap, min_gap*(1+pct)]
    """
    by_company = defaultdict(list)
    for name, amt, day in transactions:
        by_company[name].append((day, amt))
    res = []
    for name, arr in by_company.items():
        arr.sort()
        if _has_recurring_tolerant(arr, pct):
            res.append(name)
    return res
  
def _has_recurring_tolerant(day_amt_list, pct):
    n = len(day_amt_list)
    if n < 3:
        return False
    for i in range(n - 2):
        d1, a1 = day_amt_list[i]
        d2, a2 = day_amt_list[i + 1]
        d3, a3 = day_amt_list[i + 2]
        gap1 = d2 - d1
        gap2 = d3 - d2
        if gap1 <= 0 or gap2 <= 0:
            continue
        # ---- Minimal change: replace strict checks with tolerant checks ----
        if not _within_min_plus_pct([a1, a2, a3], pct):
            continue
        if not _within_min_plus_pct([gap1, gap2], pct):
            continue
        return True
    return False
  
def _within_min_plus_pct(values, pct):
    """
    Return True if every value v is within [min_val, min_val*(1+pct)].
    This matches the “min +/- 20%” (actually min to min*1.2) style
    commonly used in the Plaid recurring transaction prompt.
    """
    min_val = min(values)
    max_allowed = min_val * (1.0 + pct)
    for v in values:
        if v < min_val or v > max_allowed:
            return False
    return True


