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

  ’‘’Q1 code'''

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass(frozen=True)
class CartItem:
    price: float
    category: str


@dataclass(frozen=True)
class Coupon:
    # Q1: 只会有一个 category（面试口述常见是 valid_category）
    # 我们先用 categories[0] 来兼容后续 Q2 的多 category
    categories: List[str]

    # Exactly one of these must be non-null
    percent_discount: Optional[float]  # e.g. 20 means 20% off
    amount_discount: Optional[float]   # e.g. 6 means $6 off (one-time for that category subtotal)

    # These can be null or non-null; check against the ENTIRE cart
    minimum_num_items_required: Optional[int]
    minimum_amount_required: Optional[float]


def validate_coupon(Coupon):
    """
    Coupon is invalid if:
    - both percent_discount and amount_discount are None
    - OR both are set
    """
    c = Coupon
    has_percent = bool(c.percent_discount)
    has_amount = bool(c.amount_discount)

    # XOR: must be exactly one
    if has_percent == has_amount:
        raise ValueError("Invalid coupon: must set exactly one of percent_discount or amount_discount.")

    if not c.categories:
        raise ValueError("Invalid coupon: categories must be non-empty.")

    # Optional sanity checks
    if has_percent and c.percent_discount < 0:
        raise ValueError("Invalid coupon: percent_discount must be non-negative.")
    if has_amount and c.amount_discount < 0:
        raise ValueError("Invalid coupon: amount_discount must be non-negative.")


def cart_total_amount(cart):
    """Sum of all item prices in the cart."""
    return sum(float(it.price) for it in cart)


def cart_meets_minimums(coupon, cart):
    # Check item count minimum
    if coupon.minimum_num_items_required and len(cart) < coupon.minimum_num_items_required:
        return False
        
    # Check dollar amount minimum
    if coupon.minimum_amount_required and cart_total_amount(cart) < coupon.minimum_amount_required:
        return False
        
    return True


def build_category_subtotals(cart):
    """Compute subtotal per category for quick lookup."""
    subtotals = defaultdict(float)
    for it in cart:
        subtotals[it.category] += it.price
    return subtotals


def compute_savings_for_category(coupon, category, cat_subtotals):
  # Look up how much was spent in this specific category
    subtotal = cat_subtotals.get(category, 0)
    
    if subtotal <= 0:
        return 0
    
    # Calculate savings based on the coupon type
    if coupon.percent_discount:
        return subtotal * (coupon.percent_discount / 100)
    
    # If not percent, it's a flat amount discount
    return min(coupon.amount_discount, subtotal)


def apply_coupon_total_q1(coupon, cart):
    """
    Q1: single coupon applies to exactly one category.
    Return total after applying coupon (if minimums met).
    """
    validate_coupon(coupon)

    total = cart_total_amount(cart)

    # If minimum requirements are not met, coupon doesn't apply
    if not cart_meets_minimums(coupon, cart):
        return total

    # In Q1, we assume one category; use categories[0]
    target_category = coupon.categories[0]

    cat_subtotals = build_category_subtotals(cart)
    savings = compute_savings_for_category(coupon, target_category, cat_subtotals)

    return total - savings


# ---- Quick sanity test (matches interview experience 1, Q1) ----
if __name__ == "__main__":
    cart = [
        CartItem(2.0, "electronics"),
        CartItem(5.0, "kitchen"),
        CartItem(15.0, "food"),
    ]
    coupon = Coupon(
        categories=["electronics"],
        percent_discount=20,
        amount_discount=None,
        minimum_num_items_required=2,
        minimum_amount_required=20.0,
    )
    print(apply_coupon_total_q1(cart, coupon))  # expected 21.6

'''
Q2 变化点本质只有 3 个：

输入从 coupon 变成 coupons: List[Coupon]

每张 coupon 可以覆盖多个 categories，但只能选其中一个 category 来用 → 选 savings 最大的

多张 coupon 的 categories 不能重叠（面经的怪规则）→ 重叠直接报错
'''

from collections import Counter
from typing import Tuple


def assert_no_overlapping_categories(coupons):
    seen_categories = set()
    for c in coupons:
        for cat in c.categories:
            if cat in seen_categories:
                raise ValueError(f"Overlap detected for category: {cat}")
            seen_categories.add(cat)


def best_savings_for_coupon(coupon, cart, cat_subtotals):
    # 1. Early exit if cart doesn't meet requirements
    if not cart_meets_minimums(coupon, cart):
        return 0.0
    
    # 2. Find the one category that gives the most money back
    best = 0.0
    for cat in coupon.categories:
        savings = compute_savings(coupon, cat, cat_subtotals)
        best = max(best, savings)
        
    return best


def apply_coupons_total_q2(coupons, cart):
    # 1. Validation and Business Rules
    for c in coupons:
        validate_coupon(c)
    assert_no_overlapping_categories(coupons)
    
    # 2. Setup data
    total = sum(item.price for item in cart)
    cat_subtotals = build_category_subtotals(cart)
    
    # 3. Sum up the best savings from each coupon
    total_savings = 0.0
    for c in coupons:
        total_savings += best_savings_for_coupon(c, cart, cat_subtotals)
        
    return total - total_savings


# ---- Quick sanity test (matches interview experience 1, Q2 single coupon multi category) ----
if __name__ == "__main__":
    cart = [
        CartItem(2.0, "electronics"),
        CartItem(5.0, "kitchen"),
        CartItem(15.0, "food"),
    ]
    coupon2 = Coupon(
        categories=["electronics", "food"],
        percent_discount=20,
        amount_discount=None,
        minimum_num_items_required=2,
        minimum_amount_required=20.0,
    )
    print(apply_coupons_total_q2([coupon2], cart))  # expected 19.0

