'''
Netting Engine Central bank - ACH association
 Minimum number of transfers required at the end of the day 
 based on the net balances of individual banks
 A B C banks -> list of input transfers -> output transfers

    Central Bank Algorithm
    ----------------------

    1. Calc net balance of each bank
    2. Generate central bank transfer with either send/receive 
        the net amount to settle all other banks net balance to 0
'''

###########
''' Answer Q1 calculate balance for banks and settle with central bank '''

from collections import defaultdict

def settle_with_central_bank(transfers, central="A"):
  # clarify whether central bank itself is in the transactions
    """
    Main question:
    - transfers: list of (src, dst, amount)
    - central: central bank name (default "A")

    Return: list of (src, dst, amount) settlement transfers that net everyone to 0,
            using the central bank as the hub.
    """
    net = defaultdict(int)
    # 1) Compute net balance for each bank: incoming - outgoing
    for src, dst, amt in transfers:
        net[src] -= amt
        net[dst] += amt

    # 2) Settle via central bank
    settlements = []
    for bank, bal in net.items():
        if bank == central:
            continue
        if bal > 0:
            # bank is net receiver: central pays it
            settlements.append((central, bank, bal))
        elif bal < 0:
            # bank is net sender: it pays central
            settlements.append((bank, central, -bal))
    return settlements

### If input is like 'AB10', make small code change  to accomodate ##
def parse_transfers(encoded):
    """
    encoded: list of strings like "AB10" meaning A->B amount 10
    Assumption: bank ids are single characters, amount is integer after.
    """
    res = []
    for s in encoded:
        src = s[0]
        dst = s[1]
        amt = int(s[2:])
        res.append((src, dst, amt))
    return res

# example
encoded = ["AB1", "BC2", "CA3"]
transfers = parse_transfers(encoded)
print(settle_with_central_bank(transfers, central="A"))

''' if we need to sort the output '''
def settle_with_central_bank_sorted(transfers, central="A"):
    settlements = settle_with_central_bank(transfers, central)
    # sort by src then dst for deterministic output
    settlements.sort(key=lambda x: (x[0], x[1]))
    return settlements

''' 
Possible Q2 follow up question
Now we allow direct bank-to-bank settlement
Now you want the minimum number of transactions to settle all balances 
similar to (LeetCode 465 Optimal Account Balancing)
'''

## Reuse the same net computation
## Replace the “emit A-hub settlements” part with a DFS that minimizes count

def min_transactions_to_settle(transfers):
    """
    Follow-up: allow direct settlement between banks.
    Return the minimum number of transactions needed.
    """
    net = defaultdict(int)
    # Reuse the same net computation
    for src, dst, amt in transfers:
        net[src] -= amt
        net[dst] += amt
    balances = [bal for bal in net.values() if bal != 0]
    return _dfs_min_txn(balances, 0)

def _dfs_min_txn(balances, start):
    # skip already-settled entries
    while start < len(balances) and balances[start] == 0:
        start += 1
    if start == len(balances):
        return 0
    best = float("inf")
    # Each DFS step performs one transaction:
    # Take the first person who owes or is owed money and try to settle with someone with the opposite sign.
    # Then recursively solve the rest.
    # We try all possible pairings and choose the minimum number of transactions.
    for i in range(start + 1, len(balances)):
        if balances[start] * balances[i] < 0:
          # Save original value for backtracking
            orig_i = balances[i]
            # Simulate one transaction:
            # balances[start] is fully pushed into balances[i]
            balances[i] += balances[start] 
            best = min(best, 1 + _dfs_min_txn(balances, start + 1))
            balances[i] = orig_i  # backtrack
            # pruning: if perfect cancellation, stop exploring further i's
            if orig_i + balances[start] == 0:
                break
    return best

