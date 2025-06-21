# corp_designators.py
# Extend these lists as you find new edge-cases.

CORP_PREFIXES = [
    r"pt", r"cv", r"ud", r"sdn", r"gmbh & co",
]

CORP_SUFFIXES = [
    # Western / global
    r"inc(?:orporated)?", r"corp(?:oration)?", r"l\.?l\.?c\.?", r"l\.?l\.?p\.?",
    r"ltd\.?", r"limited", r"plc", r"co(?:\.|mpany)?\s*ltd\.?", r"gmbh",
    r"bv", r"ab", r"oy", r"s\.?a\.?r\.?l\.?", r"s\.?a\.?",

    # AU / NZ
    r"pty\.?\s*ltd\.?", r"bhd\.?",

    # SE Asia
    r"sdn\.?\s*bhd\.?", r"sendirian\s+berhad", r"pte\.?\s*ltd\.?", r"pte",

    # Indonesia
    r"tbk", r"persero", r"perseroan\s+terbatas",

    # Greater China
    r"l\.?p\.?", r"company\s+limited", r"股份有限公司", r"有限责任公司",

    # Japan
    r"kabushiki\s+kaisha", r"kabushiki\s+gaisha", r"kk", r"gk",

    # Korea
    r"주식회사",

    # India
    r"pvt\.?\s*ltd\.?", r"private\s+limited", r"llp",

    # Vietnam
    r"joint\s+stock\s+company", r"jsc", r"tnhh",

    # Thailand
    r"public\s+company\s+limited", r"pcl",

    # Philippines
    r"corporation", r"inc\.",
]
