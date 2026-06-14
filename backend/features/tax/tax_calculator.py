def calculate_taiwan_tax_2026(params: dict) -> str:
    """
    114年度（2026年申報）台灣綜合所得稅試算。
    params 必填：gross_income
    其餘未提供者皆使用合理預設值。
    """
    gross_income     = float(params["gross_income"])
    num_under_70     = int(params.get("num_under_70", 1))
    num_over_70      = int(params.get("num_over_70", 0))
    marital_status   = params.get("marital_status", "n")
    itemized         = params.get("itemized_deduction")
    salary_earners   = int(params.get("salary_earners", 1))
    rent_amount      = float(params.get("rent_amount", 0))
    savings_interest = float(params.get("savings_interest", 0))
    num_disabled     = int(params.get("num_disabled", 0))
    num_preschool    = int(params.get("num_preschool", 0))
    num_college      = int(params.get("num_college", 0))
    num_ltc          = int(params.get("num_ltc", 0))

    total_people = num_under_70 + num_over_70

    exemption = num_under_70 * 97000 + num_over_70 * 145500

    standard_deduction = 262000 if marital_status == "y" else 131000
    general_deduction  = max(standard_deduction, float(itemized)) if itemized is not None else standard_deduction

    salary_deduction   = salary_earners * 218000
    rent_deduction     = min(rent_amount, 180000)
    savings_deduction  = min(savings_interest, 270000)
    disabled_deduction = num_disabled * 218000

    preschool_deduction = 0
    if num_preschool >= 1:
        preschool_deduction += 150000
    if num_preschool > 1:
        preschool_deduction += (num_preschool - 1) * 225000

    tuition_deduction = num_college * 25000
    ltc_deduction     = num_ltc * 180000

    special_total = (salary_deduction + rent_deduction + savings_deduction +
                     disabled_deduction + preschool_deduction + tuition_deduction + ltc_deduction)

    basic_living_total = total_people * 213000
    compare_special    = (rent_deduction + savings_deduction + disabled_deduction +
                          preschool_deduction + tuition_deduction + ltc_deduction)
    basic_living_diff  = max(0, basic_living_total - exemption - general_deduction - compare_special)

    net_income = max(0, gross_income - exemption - general_deduction - special_total - basic_living_diff)

    if net_income <= 590000:
        tax_rate, prog = 0.05, 0
    elif net_income <= 1330000:
        tax_rate, prog = 0.12, 41300
    elif net_income <= 2660000:
        tax_rate, prog = 0.20, 147700
    elif net_income <= 4980000:
        tax_rate, prog = 0.30, 413700
    else:
        tax_rate, prog = 0.40, 911700

    tax_to_pay = max(0, net_income * tax_rate - prog)

    lines = [
        "📊 2026年所得稅試算（114年度）",
        "─────────────────",
        f"綜合所得總額：    ${gross_income:>12,.0f}",
        f"(-) 免稅額：      ${exemption:>12,.0f}",
        f"(-) 一般扣除額：  ${general_deduction:>12,.0f}",
        f"(-) 特別扣除額：  ${special_total:>12,.0f}",
        f"(-) 基本生活費差：${basic_living_diff:>12,.0f}",
        "─────────────────",
        f"所得淨額：        ${net_income:>12,.0f}",
        f"適用稅率：        {tax_rate * 100:.0f}%",
        "─────────────────",
        f"🧾 預估應繳稅額： ${tax_to_pay:>12,.0f}",
    ]

    if tax_rate >= 0.20 and (preschool_deduction > 0 or ltc_deduction > 0):
        lines += ["", "⚠️ 所得達20%稅率，幼兒學前／長期照顧扣除", "   因排富條款實際申報時無法適用"]

    return "\n".join(lines)
