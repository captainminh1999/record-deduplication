# corp_designators.py
# Extend these lists as you find new edge-cases.

CORP_PREFIXES = [
    r"pt",
    r"cv",
    r"ud",
    r"sdn",
    r"gmbh & co",
]

CORP_SUFFIXES = [
    # Western / global
    r"inc(?:orporated)?",
    r"corp(?:oration)?",
    r"l\.?l\.?c\.?",
    r"l\.?l\.?p\.?",
    r"ltd\.?",
    r"limited",
    r"plc",
    r"co(?:\.|mpany)?\s*ltd\.?",
    r"gmbh\b",
    r"bv",
    r"ab",
    r"oy",
    r"s\.?a\.?r\.?l\.?",
    r"s\.?a\.?\b",

    # Europe
    # Belgium / Luxembourg
    r"cvba",
    r"bvba",
    r"sab",

    # France
    r"sa\b",
    r"societe\s+anonyme",

    # Germany / Austria / Switzerland
    r"ag\b",
    r"aktiengesellschaft",
    r"gesellschaft\s+mit\s+beschrankter\s+haftung",

    # Italy
    r"spa\b",
    r"societa\s+per\s+azioni",
    r"srl\b",
    r"s\.?r\.?l\.?",
    r"societa\s+a\s+responsabilita\s+limitata",

    # Spain / Latin America
    r"s\.?a\.?\b",
    r"sociedad\s+an[oó]nima",
    r"s\.?l\.?\b",
    r"sociedad\s+limitada",
    r"srl\b",
    r"sociedad\s+de\s+responsabilidad\s+limitada",

    # Netherlands
    r"nv\b",
    r"naamloze\s+vennootschap",
    r"cv\b",
    r"commanditaire\s+vennootschap",

    # Poland
    r"sp\.?\s*z\s*o\.?\s*o\.?\b",
    r"spolka\s+z\s+ograniczona\s+odpowiedzialnoscia",
    r"sa\b",
    r"spolka\s+akcyjna",

    # Czech / Slovakia
    r"spol\s*s\s*r\.?o\.?\b",
    r"spolecnost\s+s\r?o\.?",
    r"s\.?r\.?o\.?\b",

    # Portugal / Brazil
    r"ltda\b",
    r"limitada",
    r"sociedade\s+anônima",

    # AU / NZ
    r"pty\.?\s*ltd\.?", r"bhd\.?",

    # SE Asia
    r"sdn\.?\s*bhd\.?", r"sendirian\s+berhad", r"pte\.?\s*ltd\.?", r"pte",

    # Indonesia
    r"tbk",
    r"persero",
    r"perseroan\s+terbatas",
    r"pt\.?\s*(?:persero)?\s*terbatas",
    r"perusahaan\s+terbatas",

    # Greater China
    r"l\.?p\.?",
    r"company\s+limited",
    r"股份有限公司",
    r"有限责任公司",
    r"youxian\s+gongsi",
    r"gongsi\s+youxian",

    # Japan
    r"kabushiki\s+kaisha",
    r"kabushiki\s+gaisha",
    r"kk",
    r"gk",
    r"godo\s+kaisha",

    # Korea
    r"주식회사",
    r"yuhan\s+hoesa",
    r"yh",

    # India
    r"pvt\.?\s*ltd\.?",
    r"private\s+limited",
    r"llp",
    r"esignation\s+llp",
    r"limited\s+liability\s+partnership",

    # Vietnam
    r"joint\s+stock\s+company", r"jsc", r"tnhh",

    # Thailand
    r"public\s+company\s+limited", r"pcl",

    # Philippines
    r"corporation", r"inc\.",

    # United States
    r"dba\b",
    r"d/b/a\b",
    r"doing\s+business\s+as",
    r"lllp\b",
    r"lp\b",
    r"pc\b",
    r"professional\s+corporation",

    # Canada
    r"lte\b",
    r"limited\s+by\s+shares",

    # Brazil
    r"eireli\b",
    r"empresa\s+individual\s+de\s+responsabilidade\s+limitada",

    # South Africa
    r"pty\b",
    r"proprietary\s+limited",
    r"npc\b",
    r"non-profit\s+company",

    # General non-profit / unlicensed
    r"association",
    r"foundation",
    r"trust",
    r"ngo\b",

    # Russia & CIS
    r"ооо", r"zao", r"оао", r"ao", r"тов", r"пao", r"прaт",

    # Turkey
    r"a\.ş\.", r"anonim\s+şirket", r"ltd\.\s*şti\.", r"limited\s+şirketi",

    # Israel
    r"בע""מ", r"חברה\s+בע""מ",

    # Greece
    r"α\.ε\.", r"ανώνυμη\s+εταιρεία", r"ε\.π\.ε\.", r"εταιρεία\s+περιορισμένης\s+ευθύνης",

    # Hungary
    r"kft\.", r"korlátolt\s+felelősségű\s+társaság", r"rt\.", r"részvénytársaság", r"zrt\.", r"nyrt\.",

    # Romania
    r"s\.r\.l\.", r"societate\s+cu\s+răspundere\s+limitată", r"s\.a\.", r"societate\s+pe\s+acțiuni",

    # Ukraine
    r"тов", r"tov", r"пaт", r"pat", r"прaт", r"prat",

    # Middle East
    r"pjsc", r"public\s+joint\s+stock\s+company", r"spc", r"single\s+person\s+company",

    # Nordic
    r"aps", r"a/s", r"as", r"oyj",

    # Balkans
    r"d\.o\.o\.", r"društvo\s+s\s+ograničenom\s+odgovornošću", r"a\.d\.",

    # North Africa / Francophone
    r"eurl", r"entreprise\s+unipersonnelle\s+à\s+responsabilité\s+limitée", r"snc", r"société\s+en\s+nom\s+collectif",

    # South America (additional)
    r"spa", r"sociedad\s+por\s+acciones",
]
