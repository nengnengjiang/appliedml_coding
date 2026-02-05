'''题目上下文（从面经还原的“口述版”）

你有一个购物车 cart（每个 item 有 price 和 category），以及一个或多个 coupon。

Coupon 字段（面经共识）

categories：可用的品类（第一问是单个 category；第二问是多个 category）

折扣二选一：

percent_discount：百分比折扣（比如 20 表示 20% off）

amount_discount：立减金额（比如 6 表示减 $6）

规则：必须恰好一个非空；两个都空或两个都非空 => coupon 无效，报错

使用门槛（可以为空）：

minimum_num_items_required：购物车最少件数

minimum_amount_required：购物车最少总金额

关键细节（从例子推断）：门槛是对“整个购物车”判断，不是对某个 category 子集判断
（因为例子里 coupon 门槛是 2 件 & $20，总购物车满足才打到某个 category 上。）

折扣作用范围

Coupon 只作用于一个 category（第一问固定那个 category；第二问你要选择一个最划算的 category 来用）

折扣计算等价于对该 category 的小计 cat_subtotal 做：

percent：cat_subtotal * (1 - p/100)

amount：max(0, cat_subtotal - amount)
（也可以理解为对该类商品“整体”打折/减免一次）
'''
'''
Questions

Q1（单 category）
  
  给定一个 coupon（只对应一个 category）和购物车，若 coupon 有效且门槛满足，则对该 category 打折，输出使用后的购物车总价；否则不打折（但 coupon 若字段非法要报错）。
  
  面经 1 的例子：electronics 20% off，门槛满足（购物车 3 件，总额 22），electronics 小计 2 -> 1.6，总额 21.6。

Q2（多 category + 多 coupons）

  coupon 的 categories 可能是多个，你需要选择把它用在哪一个 category 上最省钱。
  
  可能有多个 coupon。
  
  面经里有个“怪规则”：如果存在两个 coupon 都覆盖同一个 category，直接报错（不让你做“分配/冲突消解”）。
  
  面经 1 第二问：coupon 可用 [electronics, food]，门槛满足；选 food（15 * 0.8 = 12）比选 electronics 更省。

  ‘’‘

  ’‘’Q1 answer code'''

def apply_coupon_total_q1(coupon, cart):
    
  # 1. Validation (Keep it brief)
    if (coupon.percent_discount is None) == (coupon.amount_discount is None):
        raise ValueError("Must set exactly one discount type.")
    
    # 2. Scope: Calculate totals for the specific category AND total cart
    target_cat = coupon.categories[0]
    cat_subtotal = 0
    total_amount = 0
    
    for item in cart:
        total_amount += item.price
        if item.category == target_cat:
            cat_subtotal += item.price

    # 3. Check Minimums (Standard Requirement: Entire Cart)
    if coupon.minimum_num_items_required and len(cart) < coupon.minimum_num_items_required:
        return total_amount
    if coupon.minimum_amount_required and total_amount < coupon.minimum_amount_required:
        return total_amount

    # 4. Calculate Savings
    if coupon.percent_discount:
        savings = cat_subtotal * (coupon.percent_discount / 100)
    else:
        savings = min(cat_subtotal, coupon.amount_discount)
        
    return total_amount - savings


# ---- Quick Unit  test (matches interview experience 1, Q1) ----


# Simple Mock Classes (if not already defined)
class CartItem:
    def __init__(self, price, category):
        self.price = price
        self.category = category

class Coupon:
    def __init__(self, categories, percent_discount, amount_discount, min_items, min_amount):
        self.categories = categories
        self.percent_discount = percent_discount
        self.amount_discount = amount_discount
        self.minimum_num_items_required = min_items
        self.minimum_amount_required = min_amount
      

if __name__ == "__main__":
    # 1. Setup Data
    cart = [
        CartItem(2.0, "electronics"),
        CartItem(5.0, "kitchen"),
        CartItem(15.0, "food"),
    ]
    
    coupon_q1 = Coupon(
        categories=["electronics"], 
        percent_discount=20, 
        amount_discount=None, 
        min_items=2, 
        min_amount=20.0
    )

    # 2. Run and Print
    apply_coupon_total_q1(cart, coupon_q1)  # Expected: 21.6




'''
Q2 变化点本质只有 3 个：

输入从 coupon 变成 coupons: List[Coupon]

每张 coupon 可以覆盖多个 categories，但只能选其中一个 category 来用 → 选 savings 最大的

多张 coupon 的 categories 不能重叠（面经的怪规则）→ 重叠直接报错
'''

# 1. The Validation Helper (Overlapping Categories)
# Before doing any math, you need to ensure no two coupons share a category.
def assert_no_overlapping_categories(coupons):
    seen_categories = set()
    for coupon in coupons:
        for cat in coupon.categories:
            if cat in seen_categories:
                raise ValueError(f"Duplicate category found across coupons: {cat}")
            seen_categories.add(cat)

# The Requirements Check (New)
# You need this function to handle the requirement that minimums apply only to the categories listed on the coupon.
def cart_meets_minimums(coupon, cart):
    # Filter for items that belong to the coupon's categories
    matching_items = [it for it in cart if it.category in coupon.categories]
    matching_count = len(matching_items)
    matching_amount = sum(it.price for it in matching_items)

    if coupon.minimum_num_items_required and matching_count < coupon.minimum_num_items_required:
        return False
    if coupon.minimum_amount_required and matching_amount < coupon.minimum_amount_required:
        return False
    return True


def apply_coupons_total_q2(coupons, cart):
    # 1. Validation
    assert_no_overlapping_categories(coupons)
    
    # 2. Setup subtotals
    total_cart_value = sum(item.price for item in cart)
    category_subtotals = {}
    for item in cart:
        category_subtotals[item.category] = category_subtotals.get(item.category, 0) + item.price
    
    # 3. Calculate total savings
    total_savings = 0.0
    for coupon in coupons:
        if not cart_meets_minimums(coupon, cart):
            continue
            
        # Find the one category under this coupon that gives the most savings
        coupon_best = 0.0
        for cat in coupon.categories:
            subtotal = category_subtotals.get(cat, 0)
            if coupon.percent_discount:
                savings = subtotal * (coupon.percent_discount / 100)
            else:
                savings = min(subtotal, coupon.amount_discount)
            coupon_best = max(coupon_best, savings)
        
        total_savings += coupon_best
        
    return total_cart_value - total_savings


# ---- Quick sanity test (matches interview experience 1, Q2 single coupon multi category) ----

# Setup Q2 Coupons to test

    #     coupon_q1 = Coupon(
    #     categories=["electronics"], 
    #     percent_discount=20, 
    #     amount_discount=None, 
    #     min_items=2, 
    #     min_amount=20.0
      
    c_elec = Coupon(["electronics"], None, 1.0, 1, 1.0) # $1 off electronics
    c_food = Coupon(["food"], 10, None, 1, 1.0)        # 10% off food ($15 * 0.1 = 1.5)
    
    coupons = [c_elec, c_food]

    # 4. Run and Print

    # Total cart is 22.0. Total savings: 1.0 + 1.5 = 2.5. Expected: 19.5
    apply_coupons_total_q2(coupons, cart) ")# expected 19.0

