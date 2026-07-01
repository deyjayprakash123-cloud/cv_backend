import re
import io
import os
import pickle

# Try importing joblib
try:
    import joblib
except ImportError:
    joblib = None

# Try importing docx (python-docx)
try:
    from docx import Document
except ImportError:
    Document = None

# Try importing PyPDF2 or pypdf
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

def normalize_text(text: str) -> str:
    """Normalize text to resolve ligatures, smart quotes, dashes, and extra whitespace."""
    ligatures = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '–': '-',  # en dash
        '—': '-',  # em dash
        '’': "'",  # smart single quotes
        '‘': "'",
        '”': '"',  # smart double quotes
        '“': '"'
    }
    for k, v in ligatures.items():
        text = text.replace(k, v)
        
    # Replace carriage returns and normalize multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes using PdfReader."""
    if PdfReader is None:
        print("Error: PyPDF2/pypdf is not installed.")
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file bytes using python-docx."""
    if Document is None:
        print("Error: python-docx is not installed.")
        return ""
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error parsing DOCX: {e}")
        return ""

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT file bytes."""
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error parsing TXT: {e}")
        return ""

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Detect file type and extract text from bytes."""
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif ext in ['docx', 'doc']:
        return extract_text_from_docx(file_bytes)
    else:
        return extract_text_from_txt(file_bytes)

def split_sections(text: str) -> dict:
    """
    Split the resume text into Experience, Education, and Other sections
    based on common headers to avoid cross-contamination of dates and keywords.
    """
    sections = {"experience": "", "education": "", "other": ""}
    lines = text.split('\n')
    
    current_section = "other"
    
    edu_headers = ['education', 'academic', 'studies', 'qualification', 'degrees', 'university', 'college', 'schooling']
    exp_headers = ['experience', 'employment', 'work', 'history', 'career', 'professional', 'positions', 'jobs']
    
    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean:
            continue
            
        # Check if line is a section header (short line containing header keywords)
        words = line_clean.split()
        if len(words) <= 4:
            is_header = False
            for word in words:
                word_sub = re.sub(r'[^\w]', '', word)
                if word_sub in edu_headers:
                    current_section = "education"
                    is_header = True
                    break
                elif word_sub in exp_headers:
                    current_section = "experience"
                    is_header = True
                    break
            if is_header:
                continue
                
        sections[current_section] += line + "\n"
        
    return sections

def parse_date(date_str: str) -> tuple:
    """Parse a date string and return (year, month). Returns (None, None) if not parsed."""
    date_str = date_str.strip().lower()
    if not date_str:
        return None, None
        
    if date_str in ['present', 'current', 'now']:
        # Current local time from system context is July 2026
        return 2026, 7
        
    months_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    # Pattern 1: Month Name + Year (allow 4-digit or 2-digit years, e.g. "June 2018", "Jan 18")
    m1 = re.search(r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*,?\s*(\d{2,4})\b', date_str)
    if m1:
        mon_str, yr_str = m1.groups()
        month = months_map[mon_str]
        year = int(yr_str)
        if year < 100:
            year = 2000 + year if year < 70 else 1900 + year
        return year, month
        
    # Pattern 2: Numeric Month/Year (allow 4-digit or 2-digit years, e.g. "06/2018", "6/18")
    m2 = re.search(r'\b(\d{1,2})\s*[\/\-]\s*(\d{2,4})\b', date_str)
    if m2:
        mon_val = int(m2.group(1))
        yr_val = int(m2.group(2))
        if 1 <= mon_val <= 12:
            if yr_val < 100:
                yr_val = 2000 + yr_val if yr_val < 70 else 1900 + yr_val
            return yr_val, mon_val
            
    # Pattern 3: Year only (MUST be 4 digits to avoid false positives with random numbers)
    m3 = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if m3:
        return int(m3.group(1)), 1
        
    return None, None

def parse_experience(text: str) -> float:
    """
    Extract experience years:
    1. Prioritize explicit patterns like 'X years', 'X yrs', or 'X years of experience'.
    2. Fallback: Extract date ranges from the Work Experience sections (excluding Education),
       merge overlapping intervals, and sum their durations.
    """
    text_norm = normalize_text(text)
    text_lower = text_norm.lower()
    
    text_nums = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15
    }
    
    # 1. Matches explicit "8 years of experience", "10+ yrs", etc.
    explicit_pattern = re.compile(
        r'\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of\s+experience|\s+experience)?\b', 
        re.IGNORECASE
    )
    explicit_matches = explicit_pattern.findall(text_lower)
    
    years = []
    for match in explicit_matches:
        if match in text_nums:
            years.append(float(text_nums[match]))
        else:
            try:
                years.append(float(match))
            except ValueError:
                pass
                
    if years:
        return float(max(years))
        
    # 2. Date ranges fallback
    # Separate CV into sections to parse experience from work experience only (ignoring education dates)
    sections = split_sections(text)
    exp_text = sections["experience"] if sections["experience"].strip() else sections["other"]
    
    if not exp_text.strip():
        exp_text = text
        
    # Build regex patterns for date ranges
    months_pattern = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{1,2})'
    separator = r'\s*[\/\-,\s]\s*'
    
    # Range regex components:
    # Match either Month+Year or 4-digit Year to another Month+Year/4-digit Year/Present
    date_expr_with_month = rf'{months_pattern}{separator}(?:\b\d{{2,4}}\b)'
    date_expr_year_only = r'\b(?:19\d{2}|20\d{2})\b'
    date_expr = f'(?:{date_expr_with_month}|{date_expr_year_only})'
    
    range_regex = re.compile(
        rf'({date_expr})\s*(?:-|–|—|to|until)\s*({date_expr}|present|current|now)\b',
        re.IGNORECASE
    )
    
    matches = range_regex.findall(exp_text)
    ranges = []
    
    for start_str, end_str in matches:
        s_yr, s_mo = parse_date(start_str)
        e_yr, e_mo = parse_date(end_str)
        
        if s_yr is not None and e_yr is not None:
            start_date = s_yr + s_mo/12.0
            end_date = e_yr + e_mo/12.0
            if 0 < (end_date - start_date) <= 45:
                ranges.append((start_date, end_date))
                
    if not ranges:
        return 0.0
        
    # Sort and merge overlapping intervals to get true non-overlapping work duration
    ranges.sort(key=lambda x: x[0])
    merged_ranges = []
    for current in ranges:
        if not merged_ranges:
            merged_ranges.append(current)
        else:
            prev_start, prev_end = merged_ranges[-1]
            curr_start, curr_end = current
            if curr_start <= prev_end:  # Overlap
                merged_ranges[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged_ranges.append(current)
                
    total_exp = sum(end - start for start, end in merged_ranges)
    return round(total_exp, 1)

def parse_education(text: str) -> int:
    """
    Map education keywords to numeric levels (highest level wins).
    Implements case-safe checks for BE/BS/MS to avoid matching English verbs/pronouns.
    PhD/Doctorate -> 5
    MBA/M.Tech/M.Sc/Master/M.S. -> 4
    B.Tech/B.Sc/Bachelor/B.S./B.E. -> 3
    Diploma -> 2
    High School -> 1
    """
    text_norm = normalize_text(text)
    text_lower = text_norm.lower()
    
    # 5. PhD
    if re.search(r'\b(?:phd|ph\.d\.(?!\w)|ph\.d\b|doctorate)\b', text_lower):
        return 5
        
    # 4. Master's
    # Exclude MS Office, MS Excel, MS Word, etc.
    ms_exclude_pattern = r'\bms\b(?!\s+(?:office|excel|word|powerpoint|project|visio|teams|outlook|access|sql|azure|paint|windows|sharepoint|dynamics))'
    
    has_ms_dots = re.search(r'\bm\.s\.(?!\w)', text_lower) or re.search(r'\bm\.s\b', text_lower)
    has_other_masters = re.search(r'\b(?:mba|m\.tech\.(?!\w)|mtech|m\.sc\.(?!\w)|msc|masters?)\b', text_lower)
    
    is_real_ms = False
    if re.search(ms_exclude_pattern, text_lower):
        # Case-sensitive check for uppercase MS, or followed by degree context
        ms_matches = re.finditer(r'\bms\b', text_lower)
        for match in ms_matches:
            start, end = match.start(), match.end()
            orig_word = text_norm[start:end]
            context = text_lower[end:end+15]
            if orig_word == 'MS' or re.search(r'^\s+(?:in|of|degree|from)\b', context):
                is_real_ms = True
                break
                
    if has_ms_dots or has_other_masters or is_real_ms:
        return 4
        
    # 3. Bachelor's
    has_bachelors_words = re.search(r'\b(?:b\.tech\.(?!\w)|btech|b\.sc\.(?!\w)|bsc|bachelors?)\b', text_lower)
    has_bs_be_dots = re.search(r'\bb\.s\.(?!\w)', text_lower) or re.search(r'\bb\.s\b', text_lower) or re.search(r'\bb\.e\.(?!\w)', text_lower) or re.search(r'\bb\.e\b', text_lower)
    
    is_real_be_bs = False
    be_bs_matches = re.finditer(r'\b(?:be|bs)\b', text_lower)
    for match in be_bs_matches:
        start, end = match.start(), match.end()
        orig_word = text_norm[start:end]
        context = text_lower[end:end+15]
        if orig_word in ['BE', 'BS'] or re.search(r'^\s+(?:in|of|degree|from)\b', context):
            is_real_be_bs = True
            break
            
    if has_bachelors_words or has_bs_be_dots or is_real_be_bs:
        return 3
        
    # 2. Diploma
    if re.search(r'\bdiploma\b', text_lower):
        return 2
        
    # 1. High School
    if re.search(r'\bhigh\s*school\b', text_lower):
        return 1
        
    return 1

def parse_skill_count(text: str, vocabulary: list) -> int:
    """
    Count matching skills from vocabulary.
    Fallback: if count < 5, match common tech keywords.
    Cap maximum count at 50.
    """
    count = 0
    text_norm = normalize_text(text).lower()
    
    # 1. Match vocabulary skills
    if vocabulary is not None and len(vocabulary) > 0:
        for word in vocabulary:
            if not word or len(word) < 2:
                continue
            word_lower = word.lower()
            if word_lower in ['c++', 'c#', '.net']:
                pattern = re.escape(word_lower)
            else:
                pattern = r'\b' + re.escape(word_lower) + r'\b'
                
            if re.search(pattern, text_norm):
                count += 1
                
    # 2. Fallback to common tech keywords if count < 5
    if count < 5:
        fallback_keywords = [
            "python", "sql", "java", "c++", "machine learning", "deep learning", 
            "react", "node.js", "agile", "scrum", "leadership", "communication", 
            "project management", "docker", "git"
        ]
        for word in fallback_keywords:
            if word in ['c++']:
                pattern = re.escape(word)
            else:
                pattern = r'\b' + re.escape(word) + r'\b'
                
            if re.search(pattern, text_norm):
                count += 1
                
    # 3. Cap the maximum at 50
    return min(count, 50)

def extract_features_from_cv(text: str) -> tuple:
    """
    Extract structured features (experience_years, education_level, skill_count) 
    from unstructured resume text.
    """
    vocab = []
    try:
        # Load TF-IDF vocabulary from models folder
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        
        if not os.path.exists(tfidf_path):
            tfidf_path = os.path.join("models", "tfidf_vectorizer.pkl")
            
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                tfidf = pickle.load(f)
            vocab = tfidf.get_feature_names_out()
    except Exception as e:
        print(f"Warning loading vocabulary in cv_parser: {e}")
        
    experience = parse_experience(text)
    education_enc = parse_education(text)
    skill_count = parse_skill_count(text, vocab)
    
    return float(experience), int(education_enc), int(skill_count)

def comma_tokenizer(text):
    """Shared tokenizer/analyzer for TF-IDF vectorization across training and inference."""
    return [s.strip().lower() for s in str(text).split(',') if s.strip()]
