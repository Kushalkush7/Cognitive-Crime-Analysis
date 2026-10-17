"""
case_parser.py
==============
FIR Threat Assessment — RegEx + VADER NLP Intelligence Extraction Engine
with Lightweight NER Pipeline & Kanglish Normalization.

Statutory framework: Bharatiya Nyaya Sanhita (BNS) 2023
Replaces: Indian Penal Code (IPC) 1860

Migration map
-------------
  Murder    IPC 302  →  BNS §103
  Cheating  IPC 420  →  BNS §318
  Theft     IPC 379  →  BNS §303
  Snatching            BNS §304  [NEW]
  Organized Crime      BNS §111  [NEW]

Output dictionary keys are intentionally unchanged from the legacy IPC version
so that `routes.py`, `case_analysis.html`, and the JSON API remain fully
backwards-compatible without any template or caller modifications.
"""

import re
from typing import Dict, List, Any, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Dynamic SpaCy Loader with Fallback
# ---------------------------------------------------------------------------
_spacy_nlp = None
try:
    import spacy
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            _spacy_nlp = spacy.load("en_core_web_md")
        except Exception:
            _spacy_nlp = None
except ImportError:
    _spacy_nlp = None


# ---------------------------------------------------------------------------
# Module-level VADER instance (reused across calls for performance)
# ---------------------------------------------------------------------------
_vader = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Kanglish (Kannada-English) Text Normalization Rules
# ---------------------------------------------------------------------------
_KANGLISH_MAP = [
    (r'\baaropi\b', 'suspect accused'),
    (r'\baropi\b', 'suspect accused'),
    (r'\baaparadhi\b', 'offender suspect'),
    (r'\baparadhi\b', 'offender suspect'),
    (r'\browdy\s+sheeter\b', 'known offender suspect'),
    (r'\bmacchu\b', 'machete knife'),
    (r'\bmachu\b', 'machete knife'),
    (r'\bkatthi\b', 'knife blade'),
    (r'\bkathi\b', 'knife blade'),
    (r'\bchakku\b', 'knife'),
    (r'\btupaki\b', 'gun firearm'),
    (r'\bgundu\b', 'bullet cartridge'),
    (r'\bgaadi\b', 'vehicle'),
    (r'\bgadi\b', 'vehicle'),
    (r'\bvahana\b', 'vehicle'),
    (r'\brashte\b', 'road'),
    (r'\brasta\b', 'road'),
    (r'\bbeedi\b', 'street'),
    (r'\boni\b', 'lane'),
    (r'\bhathira\b', 'near'),
    (r'\bpakka\b', 'adjacent'),
]

_KANGLISH_SUFFIX_PATTERNS = [
    (r'(\b[A-Za-z0-9]+)-dalli\b', r'\1 near'),
    (r'(\b[A-Za-z0-9]+)-nalli\b', r'\1 near'),
    (r'(\b[A-Za-z0-9]+)-alli\b', r'\1 in'),
    (r'(\b[A-Za-z0-9]+)dalli\b', r'\1 near'),
    (r'(\b[A-Za-z0-9]+)nalli\b', r'\1 near'),
]


def normalize_kanglish_text(text: str) -> str:
    """
    Sanitize and normalize mixed Kannada-English (Kanglish) terms
    and postposition suffixes before entity matching.

    Parameters
    ----------
    text : str
        Raw FIR narrative text.

    Returns
    -------
    str
        Normalized text with standard English crime terminology.
    """
    if not text:
        return ""

    normalized = text
    # 1. Normalize postposition suffixes (e.g. Indiranagar-dalli -> Indiranagar near)
    for pat, repl in _KANGLISH_SUFFIX_PATTERNS:
        normalized = re.sub(pat, repl, normalized, flags=re.IGNORECASE)

    # 2. Map Kanglish lexicon terms
    for pat, repl in _KANGLISH_MAP:
        normalized = re.sub(pat, repl, normalized, flags=re.IGNORECASE)

    return normalized


def extract_narrative_paragraphs(text: str) -> List[str]:
    """
    Extracts clean, coherent paragraph blocks from the FIR narrative.
    Preserves natural section breaks, indented statements, and multi-line structures.

    Parameters
    ----------
    text : str
        Raw or cleaned FIR narrative text.

    Returns
    -------
    List[str]
        List of formatted paragraph strings.
    """
    if not text or not text.strip():
        return []

    normalized = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    raw_paras = re.split(r'\n\s*\n+', normalized)
    paragraphs = []

    for p in raw_paras:
        clean_p = p.strip()
        if clean_p:
            paragraphs.append(clean_p)

    if not paragraphs:
        paragraphs = [normalized]

    return paragraphs


def extract_narrative_statements(text: str) -> List[Dict[str, Any]]:
    """
    Splits narrative into sequential, clean statement lines for structured line-by-line display.
    Handles both multi-line FIRs and dense single-paragraph FIR narratives intelligently
    while protecting abbreviations, sections, currency, and numerical entities.

    Parameters
    ----------
    text : str
        Raw or cleaned FIR narrative text.

    Returns
    -------
    List[Dict[str, Any]]
        List of statement objects containing 'index' (1-based int) and 'text' (clean str).
    """
    if not text or not text.strip():
        return []

    raw_blocks = [line.strip() for line in text.strip().splitlines() if line.strip()]
    statements = []

    protected_tokens = [
        'No.', 'Sec.', 'Secs.', 'Rs.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Shri.',
        'approx.', 'viz.', 'i.e.', 'e.g.', 'vs.', 'Jan.', 'Feb.', 'Mar.',
        'Apr.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.',
        'BNS.', 'IPC.', 'CrPC.', 'a.k.a.', 'ref.', 'FIR No.'
    ]

    for block in raw_blocks:
        masked = block
        placeholders = {}
        for idx, tok in enumerate(protected_tokens):
            if tok.lower() in masked.lower():
                pattern = re.compile(re.escape(tok), re.IGNORECASE)
                for m in set(pattern.findall(masked)):
                    key = f"__PROT_TOK_{idx}_{len(placeholders)}__"
                    placeholders[key] = m
                    masked = masked.replace(m, key)

        decimal_matches = re.findall(r'\b\d+\.\d+\b', masked)
        for dm in set(decimal_matches):
            key = f"__PROT_NUM_{len(placeholders)}__"
            placeholders[key] = dm
            masked = masked.replace(dm, key)

        # Split on sentence ending punctuation followed by whitespace and capital/digit/quote
        parts = re.split(r'([.!?]+)\s+(?=[A-Z0-9"\'\(\[\+])', masked)

        reconstructed = []
        i = 0
        while i < len(parts):
            seg = parts[i]
            if i + 1 < len(parts) and re.match(r'^[.!?]+$', parts[i+1]):
                seg = seg + parts[i+1]
                i += 1
            if seg.strip():
                for k, v in placeholders.items():
                    seg = seg.replace(k, v)
                reconstructed.append(seg.strip())
            i += 1

        if reconstructed:
            statements.extend(reconstructed)
        else:
            for k, v in placeholders.items():
                block = block.replace(k, v)
            statements.append(block)

    return [{'index': i + 1, 'text': s} for i, s in enumerate(statements)]


# ---------------------------------------------------------------------------
# Weapon catalog
# ---------------------------------------------------------------------------
_WEAPON_CATALOG: dict[str, list[str]] = {
    'Firearms & Ballistics': [
        r'\bpistol[s]?\b', r'\brevolver[s]?\b', r'\bgun[s]?\b', r'\bhandgun[s]?\b',
        r'\bdeshi katta\b', r'\bkatta\b', r'\btamancha\b',
        r'\bcountry[- ]made (?:gun|firearm|pistol)\b',
        r'\brifle[s]?\b', r'\bak-?47\b', r'\bcarbine\b', r'\bshotgun[s]?\b',
        r'\bammunition\b', r'\bcartridge[s]?\b', r'\bbullet[s]?\b', r'\bmagazine[s]?\b',
        r'\bfired (?:rounds|shots)\b', r'\bspent shell[s]?\b'
    ],
    'Edged & Sharp Weapons': [
        r'\bknife\b', r'\bknives\b', r'\bdagger[s]?\b', r'\bmachete[s]?\b',
        r'\btalwar[s]?\b', r'\bsword[s]?\b', r'\bblade[s]?\b', r'\bcleaver[s]?\b',
        r'\brazor[s]?\b', r'\bchopper[s]?\b', r'\bswitchblade[s]?\b', r'\bkhukuri\b'
    ],
    'Blunt Force Weapons': [
        r'\biron rod[s]?\b', r'\bsteel (?:rod|pipe)[s]?\b', r'\blathi[s]?\b',
        r'\bwooden stick[s]?\b', r'\bcrowbar[s]?\b', r'\bbrass knuckles\b',
        r'\bbaseball bat[s]?\b', r'\bheavy wrench\b', r'\bblunt object\b'
    ],
    'Explosives & Hazardous Agents': [
        r'\bexplosive[s]?\b', r'\bied\b', r'\bbomb[s]?\b', r'\bgrenade[s]?\b',
        r'\bdetonator[s]?\b', r'\bdynamite\b', r'\bpetrol bomb[s]?\b', r'\bmolotov\b',
        r'\bacid\b', r'\bpoison\b', r'\bcyanide\b', r'\btoxic substance\b'
    ],
    'Tactical / Evasion Gear': [
        r'\bface mask[s]?\b', r'\bbalaclava[s]?\b', r'\bglove[s]?\b',
        r'\bfake (?:number plate|registration)\b', r'\bunmarked vehicle\b',
        r'\bsignal jammer\b', r'\bwalkie[- ]talkie\b', r'\bstolen vehicle\b'
    ]
}

# ---------------------------------------------------------------------------
# BNS Statutory Crime Matrix
# ---------------------------------------------------------------------------
_BNS_MATRIX: list[dict] = [
    {
        'category': 'Violent Crime & Bodily Offences',
        'ipc_sections': 'BNS Sec.103, BNS Sec.109, BNS Sec.115, BNS Sec.117, BNS Sec.137, BNS Sec.140, BNS Sec.64 | legacy IPC 302, 307, 323',
        'severity': 'Critical High',
        'weight': 35,
        'keywords': [
            r'\bmurder\b', r'\battempt to murder\b', r'\bhomicide\b', r'\bkilled\b',
            r'\bstabbed\b', r'\bshot\b', r'\bgrievous hurt\b', r'\bassaulted\b',
            r'\bkidnapp(?:ing|ed)\b', r'\babduct(?:ion|ed)\b', r'\brape\b',
            r'\bsexual assault\b', r'\bcriminal intimidation\b',
            r'\bhurt\b', r'\bstrangled\b', r'\bfatal\b', r'\bbleeding\b', r'\binjured\b'
        ]
    },
    {
        'category': 'Property & Organized Crime',
        'ipc_sections': 'BNS Sec.302, BNS Sec.303, BNS Sec.308, BNS Sec.309, BNS Sec.310, BNS Sec.329, BNS Sec.331 | legacy IPC 378, 379, 392',
        'severity': 'Elevated Risk',
        'weight': 22,
        'keywords': [
            r'\brobbery\b', r'\bdacoity\b', r'\btheft\b', r'\bstolen\b',
            r'\bextortion\b', r'\bburglary\b', r'\bbreak[- ]in\b',
            r'\bhouse[- ]breaking\b', r'\blooted\b', r'\btrespass\b',
            r'\barson\b', r'\bmischief by fire\b', r'\bransom\b'
        ]
    },
    {
        'category': 'Snatching',
        'ipc_sections': 'BNS Sec.304',
        'severity': 'Elevated Risk',
        'weight': 20,
        'keywords': [
            r'\bsnatched\b', r'\bsnatching\b', r'\bchain snatching\b',
            r'\bphone snatching\b', r'\bpurse snatching\b',
            r'\bgrabbed (?:mobile|phone|bag|chain|jewellery|wallet)\b',
            r'\bripped (?:chain|bag|mobile)\b',
            r'\bvehicle[- ]borne snatching\b', r'\bsnatch[- ]and[- ]run\b'
        ]
    },
    {
        'category': 'Financial & Cyber Offense',
        'ipc_sections': 'BNS Sec.316, BNS Sec.318, BNS Sec.336 | IT Act 66C/66D | legacy IPC 406, 420, 468',
        'severity': 'Moderate Risk',
        'weight': 15,
        'keywords': [
            r'\bfraud\b', r'\bcheating\b', r'\bforgery\b', r'\bembezzlement\b',
            r'\bmoney laundering\b', r'\bcyber\b', r'\bphishing\b', r'\bransomware\b',
            r'\bhacked\b', r'\bidentity theft\b', r'\bunauthorized access\b',
            r'\bfake bank\b', r'\botp fraud\b', r'\bcrypto fraud\b',
            r'\bforged document\b'
        ]
    },
    {
        'category': 'Organized Crime & Syndicate Activity',
        'ipc_sections': 'BNS Sec.111',
        'severity': 'Critical High',
        'weight': 32,
        'keywords': [
            r'\borganized crime\b', r'\bcrime syndicate\b', r'\bgang(?:ster)?\b',
            r'\bgangland\b', r'\bcriminal network\b', r'\bcriminal organization\b',
            r'\bmob\b', r'\bunderworld\b', r'\bcriminal conspiracy\b',
            r'\bhit\s+squad\b', r'\bcontract kill(?:ing|er)?\b',
            r'\bprotection racket\b', r'\bsupari\b', r'\bcartel\b',
            r'\bterror(?:ist)? link\b', r'\bmafia\b'
        ]
    },
    {
        'category': 'Contraband, Narcotics & Arms',
        'ipc_sections': 'NDPS Act 1985, Arms Act Sec.25/Sec.27',
        'severity': 'Critical High',
        'weight': 30,
        'keywords': [
            r'\bndps\b', r'\bnarcotics\b', r'\bdrugs\b', r'\bganja\b', r'\bheroin\b',
            r'\bcocaine\b', r'\bsmack\b', r'\bmethamphetamine\b', r'\bcontraband\b',
            r'\billegal arms\b', r'\bunlicensed (?:weapon|pistol|firearm)\b',
            r'\bsmuggling\b', r'\bpeddler\b', r'\bdrug syndicate\b', r'\bconsignment\b'
        ]
    },
    {
        'category': 'Public Order, Rioting & Unlawful Assembly',
        'ipc_sections': 'BNS Sec.190, Sec.191, Sec.196 | BNSS Sec.163 | legacy IPC 144, 147, 148, 149, 153A',
        'severity': 'Elevated Risk',
        'weight': 20,
        'keywords': [
            r'\brioting\b', r'\bunlawful assembly\b', r'\bmob violence\b',
            r'\bstone pelting\b', r'\bcommunal\b', r'\bvandalism\b',
            r'\bcurfew\b', r'\bpublic tranquility\b', r'\bagitator[s]?\b',
            r'\bsection 144\b'
        ]
    }
]


# ---------------------------------------------------------------------------
# Named Entity Recognition (NER) Pipeline
# ---------------------------------------------------------------------------

def extract_entities_ner(raw_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Lightweight NER pipeline using SpaCy with a robust RegEx fallback.
    Parses FIR narrative text into structured entity categories with confidence scores.

    Categories Extracted:
      - Suspects
      - Weapons Used
      - Incident Location
      - Vehicle Details

    Parameters
    ----------
    raw_text : str
        Raw or pre-normalized FIR text narrative.

    Returns
    -------
    dict
        Dictionary containing extracted entity lists with confidence scores:
        {
          'suspects': [{'entity': str, 'confidence': float}],
          'weapons_used': [{'entity': str, 'confidence': float}],
          'incident_location': [{'entity': str, 'confidence': float}],
          'vehicle_details': [{'entity': str, 'confidence': float}]
        }
    """
    if not raw_text or not raw_text.strip():
        return {
            'suspects': [],
            'weapons_used': [],
            'incident_location': [],
            'vehicle_details': []
        }

    # Normalize text prior to entity matching
    norm_text = normalize_kanglish_text(raw_text)

    suspects_set: Dict[str, float] = {}
    weapons_set: Dict[str, float] = {}
    locations_set: Dict[str, float] = {}
    vehicles_set: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. SpaCy Extraction (if model available)
    # ------------------------------------------------------------------
    if _spacy_nlp is not None:
        try:
            doc = _spacy_nlp(norm_text)
            for ent in doc.ents:
                ent_str = ent.text.strip()
                if not ent_str or len(ent_str) < 2:
                    continue

                if ent.label_ == "PERSON":
                    context_window = norm_text[max(0, ent.start_char - 30):min(len(norm_text), ent.end_char + 30)].lower()
                    if any(kw in context_window for kw in ['alias', 'accused', 'suspect', 'known as', 'a.k.a', 'aaropi', 'aropi', 'sheeter']):
                        suspects_set[ent_str] = max(suspects_set.get(ent_str, 0.0), 0.92)
                    else:
                        suspects_set[ent_str] = max(suspects_set.get(ent_str, 0.0), 0.75)

                elif ent.label_ in ("GPE", "LOC", "FAC"):
                    locations_set[ent_str] = max(locations_set.get(ent_str, 0.0), 0.88)

                elif ent.label_ in ("PRODUCT", "ORG"):
                    if any(v_kw in ent_str.lower() for v_kw in ['car', 'bike', 'auto', 'scooter', 'motorcycle', 'truck', 'vehicle', 'swift', 'pulsar', 'honda', 'ktm', 'yamaha', 'bullet']):
                        vehicles_set[ent_str] = max(vehicles_set.get(ent_str, 0.0), 0.85)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. RegEx & Rule-Based NER Engine (Primary & Fallback)
    # ------------------------------------------------------------------

    # A. Suspects RegEx
    suspect_patterns = [
        (r'\b(?:alias|a\.k\.a\.?|known as|named|accused|suspect|aaropi|aropi)\s+([A-Z][a-zA-Z0-9_\'-]+(?:\s+[A-Z][a-zA-Z0-9_\'-]+)?)', 0.95),
        (r'\b(?:suspect|accused|aaropi|aropi)\s+(?:is|was|identified as)\s+([A-Z][a-zA-Z0-9_\'-]+)', 0.90),
        (r'\b([A-Z][a-zA-Z0-9_\'-]+)\s*\((?:alias|a\.k\.a\.?)\s+([A-Z][a-zA-Z0-9_\'-]+)\)', 0.92),
    ]

    for pat, base_conf in suspect_patterns:
        matches = re.findall(pat, norm_text, flags=re.IGNORECASE)
        for m in matches:
            names = [m] if isinstance(m, str) else list(m)
            for name in names:
                clean_name = name.strip().title()
                if clean_name and len(clean_name) > 2 and clean_name.lower() not in ['suspect', 'accused', 'unknown', 'person']:
                    suspects_set[clean_name] = max(suspects_set.get(clean_name, 0.0), base_conf)

    # B. Weapons RegEx
    lower_norm = norm_text.lower()
    for category, patterns in _WEAPON_CATALOG.items():
        for pat in patterns:
            found = re.findall(pat, lower_norm)
            for match in found:
                clean_w = match.strip().title()
                conf = 0.95 if ('Firearms' in category or 'Explosives' in category) else 0.88
                weapons_set[clean_w] = max(weapons_set.get(clean_w, 0.0), conf)

    # C. Incident Location RegEx
    location_patterns = [
        (r'\b[A-Z][a-zA-Z0-9\s\.\'-]{2,25}\s+(?:Nagar|Layout|Colony|Enclave|Sector\s*\d+|Chowk|Cross|Main|Marg|Circle|Junction|Highway|Expressway|Station|Road|Street|Lane|Avenue|Flyover|Bypass|Ghat|Market|Bazaar|Puram|Ganj|Hall|Complex|Mall|Apartments|Police Station|Post|Village|Taluk|District)\b', 0.92),
        (r'\b(?:near|at|around|opposite|behind|adjacent to|in front of|off)\s+([A-Z][a-zA-Z0-9\s]{2,25}(?:[,\.\s]|$))', 0.85),
    ]

    for pat, base_conf in location_patterns:
        matches = re.findall(pat, norm_text)
        for m in matches:
            loc_str = re.sub(r'[\.,\n\r]+', '', m if isinstance(m, str) else m[0]).strip()
            if loc_str and len(loc_str) >= 3 and loc_str.lower() not in ['the', 'this', 'that', 'road', 'street']:
                locations_set[loc_str] = max(locations_set.get(loc_str, 0.0), base_conf)

    _INDIAN_HOTSPOTS = [
        'Bengaluru', 'Bangalore', 'Mumbai', 'Delhi', 'New Delhi', 'Kolkata',
        'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Mysuru', 'Mysore',
        'Hubballi', 'Belagavi', 'Mangaluru', 'Thane', 'Noida', 'Gurugram',
        'Gurgaon', 'Jaipur', 'Lucknow', 'Patna', 'Bhopal', 'Indore', 'Surat',
        'Kanpur', 'Nagpur', 'Visakhapatnam', 'Varanasi', 'Kochi', 'Coimbatore',
        'Chandigarh', 'Goa'
    ]
    for city in _INDIAN_HOTSPOTS:
        if re.search(r'\b' + re.escape(city) + r'\b', norm_text, re.IGNORECASE):
            locations_set[city] = max(locations_set.get(city, 0.0), 0.96)

    # D. Vehicle Details RegEx
    vehicle_reg_matches = re.findall(r'\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{4}\b', norm_text)
    for v_reg in vehicle_reg_matches:
        vehicles_set[v_reg] = max(vehicles_set.get(v_reg, 0.0), 0.98)

    vehicle_phrase_matches = re.findall(
        r'\b(?:stolen|unmarked|suspect|white|black|red|blue|grey|dark)?\s*'
        r'(?:swift|pulsar|auto|autorickshaw|car|bike|motorcycle|scooter|ktm|yamaha|honda|activa|bullet|bolero|innova|scorpio|gadi|vehicle)'
        r'(?:\s+bearing\s+number\s+[A-Z0-9\-\s]+)?\b',
        norm_text, flags=re.IGNORECASE
    )
    for v_phrase in vehicle_phrase_matches:
        clean_v = v_phrase.strip().title()
        if clean_v and len(clean_v) > 2 and clean_v.lower() not in ['the vehicle', 'a vehicle']:
            vehicles_set[clean_v] = max(vehicles_set.get(clean_v, 0.0), 0.86)

    # ------------------------------------------------------------------
    # 3. Format Output Lists with Confidence Scores
    # ------------------------------------------------------------------
    def _format_entity_list(entity_dict: Dict[str, float]) -> List[Dict[str, Any]]:
        return [
            {'entity': k, 'confidence': round(v, 2)}
            for k, v in sorted(entity_dict.items(), key=lambda item: item[1], reverse=True)
        ]

    return {
        'suspects': _format_entity_list(suspects_set),
        'weapons_used': _format_entity_list(weapons_set),
        'incident_location': _format_entity_list(locations_set),
        'vehicle_details': _format_entity_list(vehicles_set),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_case_intelligence(narrative_text: str, source_label: str = "Direct Narrative") -> dict | None:
    """
    NLP & Regex Intelligence Extraction Engine (BNS edition with NER pipeline).

    Performs:
      - Kanglish text normalization
      - VADER Sentiment analysis & threat index scoring
      - Multi-vector Entity Extraction & Lightweight NER Pipeline
      - Holistic threat classification and officer safety action protocol

    Parameters
    ----------
    narrative_text : str
        Raw FIR / case text (plain text, already extracted from PDF/TXT).
    source_label : str
        Human-readable tag describing how the text was obtained.

    Returns
    -------
    dict | None
        Structured intelligence dossier, or None if input is blank.
        All legacy output keys are maintained for full backwards compatibility.
    """
    if not narrative_text or not narrative_text.strip():
        return None

    # Step 0: Kanglish text normalization
    normalized_narrative = normalize_kanglish_text(narrative_text.strip())

    clean_text = narrative_text.strip()
    words      = clean_text.split()
    word_count = len(words)
    char_count = len(clean_text)

    # ------------------------------------------------------------------
    # 1. VADER Sentiment Analysis
    # ------------------------------------------------------------------
    sentiment  = _vader.polarity_scores(normalized_narrative)
    compound   = sentiment['compound']
    neg_score  = sentiment['neg']
    neu_score  = sentiment['neu']
    pos_score  = sentiment['pos']

    lower_text = normalized_narrative.lower()

    # ------------------------------------------------------------------
    # 2. Weapons & Threat Indicators
    # ------------------------------------------------------------------
    extracted_weapons  = []
    weapon_risk_points = 0

    for category, patterns in _WEAPON_CATALOG.items():
        found: set[str] = set()
        for pat in patterns:
            for m in re.findall(pat, lower_text):
                found.add(m.strip().title())
        if found:
            if 'Firearms' in category or 'Explosives' in category:
                risk_weight = 30
            elif 'Edged' in category:
                risk_weight = 20
            else:
                risk_weight = 12
            weapon_risk_points += len(found) * risk_weight
            extracted_weapons.append({
                'category':   category,
                'elements':   sorted(found),
                'items':      sorted(found),          # backwards-compat alias
                'badge_class': (
                    'danger'  if ('Firearms' in category or 'Explosives' in category) else
                    'warning' if 'Edged' in category else
                    'info'
                )
            })

    # ------------------------------------------------------------------
    # 3. Dates & Timeline Extraction
    # ------------------------------------------------------------------
    date_matches = re.findall(
        r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
        clean_text
    )
    named_dates = re.findall(
        r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?'
        r'(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
        clean_text, re.IGNORECASE
    )
    named_dates_alt = re.findall(
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+'
        r'(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'(?:,?\s+\d{4})?\b',
        clean_text, re.IGNORECASE
    )
    time_matches = re.findall(
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|hrs|hours)?\b',
        clean_text
    )
    mil_time = re.findall(r'\b\d{4}\s*(?:hrs|hours)\b', clean_text, re.IGNORECASE)
    context_times = re.findall(
        r"\b(?:around\s+\d{1,2}\s*(?:am|pm|o'clock)|midnight|noon|dawn|dusk|"
        r"early morning|late night|yesterday evening|last night)\b",
        lower_text
    )

    all_timeline = list(dict.fromkeys(
        date_matches + named_dates + named_dates_alt +
        time_matches + mil_time + [t.title() for t in context_times]
    ))
    timeline_signals = all_timeline[:12]

    # ------------------------------------------------------------------
    # 4. Locations & Geospatial Indicators
    # ------------------------------------------------------------------
    address_patterns = re.findall(
        r'\b[A-Z][a-zA-Z0-9\s\.\'-]{2,25}\s+'
        r'(?:Nagar|Layout|Colony|Enclave|Sector\s*\d+|Chowk|Cross|Main|Marg|'
        r'Circle|Junction|Highway|Expressway|Station|Road|Street|Lane|Avenue|'
        r'Flyover|Bypass|Ghat|Market|Bazaar|Puram|Ganj|Hall|Complex|Mall|'
        r'Apartments|Police Station|Post|Village|Taluk|District)\b',
        clean_text
    )
    preposition_locs = re.findall(
        r'\b(?:near|at|around|opposite|behind|adjacent to|in front of|off)\s+'
        r'([A-Z][a-zA-Z0-9\s]{2,25}(?:[,\.\s]|$))',
        clean_text
    )
    _INDIAN_HOTSPOTS = [
        'Bengaluru', 'Bangalore', 'Mumbai', 'Delhi', 'New Delhi', 'Kolkata',
        'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Mysuru', 'Mysore',
        'Hubballi', 'Belagavi', 'Mangaluru', 'Thane', 'Noida', 'Gurugram',
        'Gurgaon', 'Jaipur', 'Lucknow', 'Patna', 'Bhopal', 'Indore', 'Surat',
        'Kanpur', 'Nagpur', 'Visakhapatnam', 'Varanasi', 'Kochi', 'Coimbatore',
        'Chandigarh', 'Goa'
    ]
    city_matches = [
        city for city in _INDIAN_HOTSPOTS
        if re.search(r'\b' + re.escape(city) + r'\b', clean_text, re.IGNORECASE)
    ]

    combined_locs: list[str] = []
    for loc in address_patterns + preposition_locs + city_matches:
        cleaned = re.sub(r'[.,\n\r]+', '', loc).strip()
        if len(cleaned) >= 3 and cleaned not in combined_locs:
            combined_locs.append(cleaned)
    location_list = combined_locs[:10]

    # ------------------------------------------------------------------
    # 5. BNS Crime Categories & Statutory Classification
    # ------------------------------------------------------------------
    detected_ipc    = []   # key kept as 'ipc_categories' in output for compat
    ipc_risk_points = 0

    for entry in _BNS_MATRIX:
        matched_terms: set[str] = set()
        for kw in entry['keywords']:
            for m in re.findall(kw, lower_text):
                matched_terms.add(m.strip().title())
        if matched_terms:
            ipc_risk_points += entry['weight'] + (len(matched_terms) * 4)
            detected_ipc.append({
                'category':     entry['category'],
                'ipc_sections': entry['ipc_sections'],   # now contains BNS refs
                'severity':     entry['severity'],
                'matches':      sorted(matched_terms),
                'badge_class': (
                    'danger'  if 'Critical' in entry['severity'] else
                    'warning' if 'Elevated' in entry['severity'] else
                    'info'
                )
            })

    # ------------------------------------------------------------------
    # 6. Tactical Signals: Vehicles, Phones, Aliases
    # ------------------------------------------------------------------
    tactical_signals = {
        'vehicles': list(dict.fromkeys(
            re.findall(r'\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{4}\b', clean_text)
        ))[:5],
        'phone_numbers': list(dict.fromkeys(
            re.findall(r'\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b', clean_text)
        ))[:5],
        'suspect_aliases': list(dict.fromkeys(
            re.findall(
                r'\b(?:alias|a\.k\.a\.?|known as)\s+([A-Z][a-zA-Z0-9_\'-]+)',
                clean_text, re.IGNORECASE
            )
        ))[:5]
    }

    # ------------------------------------------------------------------
    # 7. Lightweight NER Pipeline Integration
    # ------------------------------------------------------------------
    ner_entities = extract_entities_ner(clean_text)

    # ------------------------------------------------------------------
    # 8. Holistic Threat Severity Rating Calculation
    # ------------------------------------------------------------------
    neg_factor        = neg_score * 85.0
    sentiment_penalty = 20.0 if compound <= -0.4 else (10.0 if compound <= -0.1 else 0.0)
    raw_threat_index  = (
        neg_factor +
        sentiment_penalty +
        min(weapon_risk_points, 35) +
        min(ipc_risk_points,    35)
    )
    threat_score = min(100, max(5, int(raw_threat_index)))

    has_critical_weapons = any(
        ('Firearms' in w['category'] or 'Explosives' in w['category'] or 'Edged' in w['category'])
        for w in extracted_weapons
    )
    has_violent_crime = any(
        ('Violent' in c['category'] or 'Contraband' in c['category'] or 'Organized' in c['category'])
        for c in detected_ipc
    )

    if threat_score >= 60 or compound <= -0.45 or (has_critical_weapons and has_violent_crime):
        severity_label = "HIGH RISK"
        severity_class = "danger"
        severity_desc  = (
            "Immediate tactical threat detected. Multiple lethal risk vectors, "
            "violent offense markers, or heightened negative polarity."
        )
        action_protocol = [
            "Mandate 2+ Armed Tactical Field Officers on approach.",
            "Dispatch Forensic Crime Scene Unit for ballistic / biological evidence retrieval.",
            "Secure surrounding CCTV telemetry and establish territorial cordon.",
            "Issue emergency lookout circular for identified suspect markers."
        ]
    elif threat_score >= 30 or compound <= -0.05 or extracted_weapons or detected_ipc:
        severity_label = "MODERATE THREAT"
        severity_class = "warning"
        severity_desc  = (
            "Elevated risk incident. Property, cyber, or physical confrontation "
            "markers detected with moderate threat indicators."
        )
        action_protocol = [
            "Deploy standard Field Patrol Unit with body-worn cameras.",
            "Verify witness testimonies and cross-examine geolocation telemetry.",
            "Preserve chain of custody for digital/physical evidence.",
            "Log suspect vehicle / contact identifiers into regional surveillance registry."
        ]
    else:
        severity_label = "LOW THREAT"
        severity_class = "safe"
        severity_desc  = (
            "Informational or low-threat occurrence. No active lethal weapons "
            "or critical violence indicators identified."
        )
        action_protocol = [
            "Log incident statement into standard non-cognizable / general diary records.",
            "Conduct routine community follow-up if applicable.",
            "Archive telemetry for retrospective analytical pattern modeling."
        ]

    excerpt = clean_text[:8000] + ('...' if len(clean_text) > 8000 else '')
    narrative_paragraphs = extract_narrative_paragraphs(clean_text)
    narrative_statements = extract_narrative_statements(clean_text)

    # ------------------------------------------------------------------
    # 9. Return structured dossier
    # ------------------------------------------------------------------
    return {
        'status':         'success',
        'algorithm_used': 'VADER NLP + RegEx Weighted BNS Severity Scoring',
        'algorithm_info': {
            'name':                'VADER NLP + RegEx Weighted BNS Severity Scoring',
            'nlp_engine':          'VADER SentimentIntensityAnalyzer',
            'statutory_framework': 'BNS 2023 (Bharatiya Nyaya Sanhita), NDPS Act 1985, Arms Act 1959',
            'scoring_methodology': 'Weighted Compound Polarity & Lethal Vector Index'
        },
        'source_label': source_label,
        'word_count':   word_count,
        'char_count':   char_count,
        'sentiment': {
            'compound': round(compound, 4),
            'neg':      round(neg_score, 3),
            'neu':      round(neu_score, 3),
            'pos':      round(pos_score, 3),
        },
        'threat_score':         threat_score,
        'severity_label':       severity_label,
        'severity_class':       severity_class,
        'severity_desc':        severity_desc,
        'action_protocol':      action_protocol,
        'weapons':              extracted_weapons,
        'locations':            location_list,
        'timeline':             timeline_signals,
        'ipc_categories':       detected_ipc,
        'tactical_signals':     tactical_signals,
        'ner_entities':         ner_entities,
        'narrative_paragraphs': narrative_paragraphs,
        'narrative_statements': narrative_statements,
        'narrative_lines':      [s['text'] for s in narrative_statements],
        'raw_excerpt':          excerpt
    }



