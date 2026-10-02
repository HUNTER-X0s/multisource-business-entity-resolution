import re, unicodedata, sys
sys.path.insert(0, '.')
from scratch.inspect_missed import missed

def get_house_no(addr):
    m = re.search(r'\b(\d{1,6}[a-zA-Z]?)\b', str(addr or ''))
    return m.group(1).lower() if m else ""

def get_first_word(nm):
    toks = [t for t in re.sub(r'[^\w\s]', ' ', str(nm or '').lower()).split() if len(t) >= 3]
    return toks[0] if toks else ""

def get_city(addr):
    # Usually the token before state / country or after comma
    parts = [p.strip().lower() for p in str(addr or '').split(',') if p.strip()]
    if len(parts) >= 2:
        return parts[-1] if len(parts[-1].split()) <= 2 else parts[-2]
    return ""

def char_3gram_jaccard(a, b):
    if not a or not b: return 0.0
    sa = {a[i:i+3] for i in range(len(a)-2)}
    sb = {b[i:i+3] for i in range(len(b)-2)}
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

recovered_hno_city = 0
recovered_name_city = 0
recovered_fuzzy_name = 0
recovered_any = 0

for q, t in missed:
    q_nm = str(q['business_name'] or '').lower()
    t_nm = str(t['business_name'] or '').lower()
    q_addr = str(q['business_address'] or '').lower()
    t_addr = str(t['business_address'] or '').lower()
    
    q_hno, t_hno = get_house_no(q_addr), get_house_no(t_addr)
    q_fw, t_fw = get_first_word(q_nm), get_first_word(t_nm)
    
    # Fuzzy name
    sim = char_3gram_jaccard(q_nm, t_nm)
    
    match = False
    if q_hno and t_hno and q_hno == t_hno and len(q_hno) >= 2:
        recovered_hno_city += 1
        match = True
    elif q_fw and t_fw and q_fw == t_fw and len(q_fw) >= 4:
        recovered_name_city += 1
        match = True
    elif sim >= 0.50:
        recovered_fuzzy_name += 1
        match = True
        
    if match:
        recovered_any += 1

print(f"Total previously missed: {len(missed)}")
print(f"Recovered by House Number:      {recovered_hno_city}")
print(f"Recovered by First Word of Name: {recovered_name_city}")
print(f"Recovered by Fuzzy Name (>=0.5): {recovered_fuzzy_name}")
print(f"Total Recovered:                 {recovered_any} / {len(missed)} ({100*recovered_any/len(missed):.2f}%)")
print(f"Remaining Still Missed:          {len(missed) - recovered_any} / 18,242 total true links")
print(f"NEW OVERALL RECALL CEILING:      {(18242 - (len(missed) - recovered_any)) / 18242 * 100:.2f}%!")
