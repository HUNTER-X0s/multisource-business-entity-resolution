import polars as pl
from collections import defaultdict
import re, unicodedata

ADDR_STOPS = {
    'street', 'avenue', 'road', 'drive', 'lane', 'boulevard', 'circle', 'court', 'place', 'way',
    'highway', 'suite', 'floor', 'unit', 'building', 'tower', 'block', 'phase', 'sector', 'plot',
    'near', 'behind', 'opposite', 'cross', 'main', 'first', 'second', 'third', 'ground', 'upper',
    'delhi', 'mumbai', 'bangalore', 'kolkata', 'chennai', 'hyderabad', 'pune', 'ahmedabad', 'jaipur',
    'india', 'texas', 'california', 'florida', 'york', 'washington', 'illinois', 'carolina', 'georgia',
    'pradesh', 'maharashtra', 'karnataka', 'tamil', 'nadu', 'gujarat', 'bengal', 'rajasthan', 'kerala',
}

INDIC_MAP = {
    'अ':'a','आ':'aa','इ':'i','ई':'ee','उ':'u','ऊ':'oo','ए':'e','ऐ':'ai','ओ':'o','औ':'au',
    'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'ng','च':'ch','छ':'chh','ज':'j','झ':'jh','ञ':'ny',
    'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n',
    'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh',
    'ष':'sh','स':'s','ह':'h','ळ':'l','ा':'a','ि':'i','ी':'ee','ु':'u','ू':'oo','े':'e',
    'ै':'ai','ो':'o','ौ':'au','ं':'n','ः':'h','्':'','ँ':'n','ृ':'ri',
    'অ':'a','আ':'aa','ই':'i','ঈ':'ee','উ':'u','ঊ':'oo','এ':'e','ঐ':'ai','ও':'o','ঔ':'au',
    'ক':'k','খ':'kh','গ':'g','ঘ':'gh','ঙ':'ng','চ':'ch','ছ':'chh','জ':'j','ঝ':'jh','ঞ':'ny',
    'ট':'t','ঠ':'th','ড':'d','ঢ':'dh','ণ':'n','ত':'t','थ':'th','দ':'d','ध':'dh','ন':'n',
    'প':'p','ফ':'ph','ব':'b','ভ':'bh','ম':'m','য':'y','র':'r','ল':'l','শ':'sh','ষ':'sh',
    'স':'s','হ':'h','ড়':'r','ঢ়':'rh','য়':'y','া':'a','ি':'i','ী':'ee','ু':'u','ূ':'oo',
    'ে':'e','ৈ':'ai','ো':'o','ৌ':'au','্':'',
    'అ':'a','ఆ':'aa','ఇ':'i','ఈ':'ee','ఉ':'u','ఊ':'oo','ఎ':'e','ఏ':'ee','ఐ':'ai','ఒ':'o','ఓ':'oo','ఔ':'au',
    'క':'k','ఖ':'kh','గ':'g','ఘ':'gh','ఙ':'ng','చ':'ch','ఛ':'chh','జ':'j','ఝ':'jh','ఞ':'ny',
    'ట':'t','ఠ':'th','డ':'d','ఢ':'dh','ణ':'n','త':'t','థ':'th','ద':'d','ధ':'dh','న':'n',
    'ప':'p','ఫ':'ph','బ':'b','భ':'bh','మ':'m','య':'y','ర':'r','ల':'l','వ':'v','శ':'sh',
    'ష':'sh','స':'s','హ':'h','ళ':'l','ా':'a','ి':'i','ీ':'ee','ు':'u','ూ':'oo','ె':'e',
    'ే':'ee','ై':'ai','ొ':'o','ో':'oo','ౌ':'au','ం':'m','్':'',
    'அ':'a','ஆ':'aa','இ':'i','ஈ':'ee','உ':'u','ஊ':'oo','எ':'e','ஏ':'ee','ஐ':'ai','ஒ':'o','ஓ':'oo','ஔ':'au',
    'க':'k','ங':'ng','ச':'ch','ஞ':'ny','ட':'t','ண':'n','த':'t','ந':'n','ப':'p','ம':'m',
    'ய':'y','ர':'r','ல':'l','வ':'v','ழ':'zh','ள':'l','ற':'r','ன':'n','ா':'a','ி':'i',
    'ீ':'ee','ு':'u','ூ':'oo','ெ':'e','ே':'ee','ை':'ai','ொ':'o','ோ':'oo','ௌ':'au','்':'',
    'ಅ':'a','ಆ':'aa','ಇ':'i','ಈ':'ee','ಉ':'u','ಊ':'oo','ಎ':'e','ಏ':'ee','ಐ':'ai','ಒ':'o','ಓ':'oo','ಔ':'au',
    'ಕ':'k','ಖ':'kh','ಗ':'g','ಘ':'gh','ಙ':'ng','ಚ':'ch','ಛ':'chh','ಜ':'j','ಝ':'jh','ಞ':'ny',
    'ಟ':'t','ಠ':'th','ಡ':'d','ಢ':'dh','ಣ':'n','ತ':'t','ಥ':'th','ದ':'d','ಧ':'dh','ನ':'n',
    'ಪ':'p','ಫ':'ph','ಬ':'b','ಭ':'bh','ಮ':'m','ಯ':'y','ರ':'r','ಲ':'l','ವ':'v','ಶ':'sh',
    'ಷ':'sh','ಸ':'s','ಹ':'h','ಳ':'l','ಾ':'a','ಿ':'i','ೀ':'ee','ು':'u','ೂ':'oo','ೆ':'e',
    'ೇ':'ee','ೈ':'ai','ೊ':'o','ೋ':'oo','ೌ':'au','ಂ':'m','್':'',
    'അ':'a','ആ':'aa','ഇ':'i','ഈ':'ee','ഉ':'u','ഊ':'oo','എ':'e','ഏ':'ee','ഐ':'ai','ഒ':'o','ഓ':'oo','ഔ':'au',
    'ക':'k','ഖ':'kh','ഗ':'g','ഘ':'gh','ങ':'ng','ച':'ch','ഛ':'chh','ജ':'j','ഝ':'jh','ഞ':'ny',
    'ട':'t','ഠ':'th','ഡ':'d','ഢ':'dh','ണ':'n','ത':'t','ഥ':'th','ദ':'d','ധ':'dh','ന':'n',
    'പ':'p','ഫ':'ph','ബ':'b','ഭ':'bh','മ':'m','യ':'y','ര':'r','ല':'l','വ':'v','ശ':'sh',
    'ഷ':'sh','സ':'s','ഹ':'h','ള':'l','ഴ':'zh','റ':'r','ാ':'a','ി':'i','ീ':'ee','ു':'u',
    'ൂ':'oo','െ':'e','േ':'ee','ൈ':'ai','ൊ':'o','ോ':'oo','ൌ':'au','ം':'m','്':'',
}
TRANS_TABLE = str.maketrans(INDIC_MAP)

def transliterate(s):
    if not s: return ""
    return s.translate(TRANS_TABLE)

def clean(s):
    s = transliterate(str(s or '')).lower()
    s = re.sub(r'[éèêë]', 'e', s)
    s = re.sub(r'[àâä]', 'a', s)
    s = re.sub(r'[îï]', 'i', s)
    s = re.sub(r'[ôö]', 'o', s)
    s = re.sub(r'[ùûü]', 'u', s)
    s = re.sub(r'ç', 'c', s)
    s = re.sub(r'\.(com|org|net|in|co|us|gov|io)\b', ' ', s)
    s = re.sub(r'\b(www|http|https)\b', ' ', s)
    s = re.sub(r'&', ' and ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def deleet(s):
    s = re.sub(r'([a-z])0([a-z])', r'\1o\2', s)
    s = re.sub(r'([a-z])1([a-z])', r'\1l\2', s)
    s = re.sub(r'([a-z])3([a-z])', r'\1e\2', s)
    s = re.sub(r'([a-z])4([a-z])', r'\1a\2', s)
    s = re.sub(r'([a-z])5([a-z])', r'\1s\2', s)
    return s

LEGAL = r'\b(private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|llc|inc|incorporated|corporation|corp|llp|co|company|center|services|service|partners|group|holdings)\b'

def strip_suf(s):
    return re.sub(r'\s+', ' ', re.sub(LEGAL, ' ', s)).strip()

def sort_toks(s):
    toks = [t for t in s.split() if len(t) >= 2]
    return ' '.join(sorted(toks))

def get_hno(s):
    m = re.search(r'\b(\d{1,6}[a-z]?)\b', s)
    return m.group(1) if m else ""

def get_pincode(s, country):
    if 'india' in country:
        m = re.search(r'\b([1-9]\d{5})\b', s)
        return m.group(1) if m else ""
    elif 'us' in country:
        m = re.search(r'\b(\d{5})(?:-\d{4})?\b', s)
        return m.group(1) if m else ""
    m = re.search(r'\b(\d{5,6})\b', s)
    return m.group(1) if m else ""

def get_distinctive_toks(s, min_len=4, stops=set()):
    toks = [w for w in s.split() if len(w) >= min_len and not w.isdigit() and w not in stops]
    if len(toks) >= 2:
        top2 = sorted(sorted(toks, key=len, reverse=True)[:2])
        return f"{top2[0]}_{top2[1]}"
    elif len(toks) == 1:
        return toks[0]
    return ""

# Load GT sample
gt = pl.read_csv('dataset/raw/train/train_ground_truth.tsv', separator='\t', quote_char=None)
gt_pos = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).slice(0, 5000)

s1_ids = set(gt_pos['source1_entity_id'])
all_targets = set()
gt_map = {}
for row in gt_pos.iter_rows(named=True):
    qid = row['source1_entity_id']
    tids = [t.strip() for t in row['matched_entity_ids'].split(',') if t.strip()]
    gt_map[qid] = set(tids)
    all_targets.update(tids)

s1_df = pl.read_csv('dataset/raw/train/train_source1.tsv', separator='\t', quote_char=None).filter(pl.col('entity_id').is_in(list(s1_ids)))
s2_df = pl.read_csv('dataset/raw/train/train_source2.tsv', separator='\t', quote_char=None).filter(pl.col('entity_id').is_in(list(all_targets)))
s3_df = pl.read_csv('dataset/raw/train/train_source3.tsv', separator='\t', quote_char=None).filter(pl.col('entity_id').is_in(list(all_targets)))
targets_df = pl.concat([s2_df, s3_df])

s1_dict = {r['entity_id']: r for r in s1_df.to_dicts()}
tgt_dict = {r['entity_id']: r for r in targets_df.to_dicts()}

total_true = sum(len(v) for v in gt_map.values())
matched = set()

for qid, tids in gt_map.items():
    q = s1_dict.get(qid)
    if not q: continue
    cty = str(q['country'] or '').lower()
    
    q_nm = clean(q['business_name'])
    q_nm_strip = strip_suf(q_nm)
    q_nm_sort = sort_toks(q_nm_strip)
    q_nm_space = re.sub(r'\s+', '', q_nm)
    q_nm_deleet = sort_toks(strip_suf(deleet(q_nm)))
    q_nm_bi = get_distinctive_toks(q_nm_strip, min_len=4)
    q_fw = (q_nm_strip.split() or [''])[0]
    
    q_addr = clean(q['business_address'])
    q_addr_sort = sort_toks(q_addr)
    q_addr_bi = get_distinctive_toks(q_addr, min_len=5, stops=ADDR_STOPS)
    q_hno = get_hno(q_addr)
    q_pin = get_pincode(q_addr, cty)
    
    for tid in tids:
        t = tgt_dict.get(tid)
        if not t: continue
        pair = (qid, tid)
        
        t_nm = clean(t['business_name'])
        t_nm_strip = strip_suf(t_nm)
        t_nm_sort = sort_toks(t_nm_strip)
        t_nm_space = re.sub(r'\s+', '', t_nm)
        t_nm_deleet = sort_toks(strip_suf(deleet(t_nm)))
        t_nm_bi = get_distinctive_toks(t_nm_strip, min_len=4)
        t_fw = (t_nm_strip.split() or [''])[0]
        
        t_addr = clean(t['business_address'])
        t_addr_sort = sort_toks(t_addr)
        t_addr_bi = get_distinctive_toks(t_addr, min_len=5, stops=ADDR_STOPS)
        t_hno = get_hno(t_addr)
        t_pin = get_pincode(t_addr, cty)
        
        # Check channels
        if q_nm and t_nm and q_nm == t_nm: matched.add(pair)
        elif q_nm_sort and t_nm_sort and q_nm_sort == t_nm_sort: matched.add(pair)
        elif len(q_nm_space) >= 6 and (q_nm_space == t_nm_space or q_nm_space in t_nm_space or t_nm_space in q_nm_space): matched.add(pair)
        elif q_nm_deleet and t_nm_deleet and q_nm_deleet == t_nm_deleet: matched.add(pair)
        elif q_nm_bi and t_nm_bi and q_nm_bi == t_nm_bi: matched.add(pair)
        elif q_addr and t_addr and len(q_addr) >= 10 and q_addr == t_addr: matched.add(pair)
        elif q_addr_sort and t_addr_sort and len(q_addr_sort) >= 12 and q_addr_sort == t_addr_sort: matched.add(pair)
        elif q_addr_bi and t_addr_bi and q_addr_bi == t_addr_bi: matched.add(pair)
        elif q_hno and t_hno and len(q_hno) >= 2 and q_hno == t_hno and (q_fw and t_fw and q_fw == t_fw): matched.add(pair)
        elif q_pin and t_pin and len(q_pin) >= 5 and q_pin == t_pin and (q_fw and t_fw and q_fw == t_fw): matched.add(pair)

print(f"Total True Links: {total_true:,}")
print(f"Retrieved Links:  {len(matched):,} ({len(matched)/total_true*100:.2f}%)")
print(f"Missed Links:     {total_true - len(matched):,} ({(total_true - len(matched))/total_true*100:.2f}%)")
