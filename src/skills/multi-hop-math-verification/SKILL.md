---
name: multi-hop-math-verification
description: Always compute multi-step arithmetic through python_interpreter with explicit intermediate values shown, instead of doing mental math. Use for any task involving more than one arithmetic operation, currency conversion, percentage calculation, or unit conversion. Prevents silent rounding and order-of-operations errors that are common in LLM mental math.
metadata:
  consumer: deep_analyzer_agent
  skill_type: verification_pattern
  source: seeded
  verified_uses: 0
  confidence: 0.9
---

# Multi-hop Math Verification (deep_analyzer_agent workflow)

## When to activate
- Task involves ≥ 2 arithmetic operations composed together, OR
- Task mentions currency / percentages / unit conversion, OR
- Task asks for "how much" / "difference between" / "ratio of" / "average of" with numeric inputs.

## Workflow
1. Identify all numeric inputs and their units (dollars, percent, meters, etc.).
2. Use `python_interpreter_tool` with explicit variable assignments — one input per line, names that reflect their role:
   ```python
   revenue_q3_usd = 90.3e9
   revenue_q2_usd = 85.7e9
   growth_pct = (revenue_q3_usd - revenue_q2_usd) / revenue_q2_usd * 100
   print(f"Growth: {growth_pct:.2f}%")
   ```
3. Show every intermediate value on its own line so the arithmetic chain is auditable.
4. After computing, validate the result against a sanity check:
   - Is the sign correct? (profits positive, losses negative)
   - Is the order of magnitude plausible? (a company's Q3 revenue is not 1e15)
   - Does the unit match what the question asked for?
5. If the inputs come from different currencies/units, do the conversion BEFORE the arithmetic, not after. Mixing USD and EUR then multiplying by a growth factor is a common failure mode.

## Avoid
- Mental math for anything beyond single-digit arithmetic. Your float intuition is bad above 4 digits.
- Reporting a number without the units. "The answer is 90" is ambiguous; "The answer is $90 billion" is not.
- Computing percentages as `new - old / old` without parentheses around the subtraction (order-of-operations trap).

## Verification example
```python
# Question: "What is the 12% VAT on $1,234.56?"
price = 1234.56
vat_rate = 0.12
vat = price * vat_rate
total = price + vat
print(f"Price: ${price:.2f}")
print(f"VAT (12%): ${vat:.2f}")
print(f"Total: ${total:.2f}")
# Sanity: vat should be ~= price / 8, total should be ~= 1.12 * price
assert abs(vat - price/8) < 10
assert abs(total - 1.12 * price) < 0.01
```
