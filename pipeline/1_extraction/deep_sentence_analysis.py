"""
Deep analysis of sentence-level dataset: quality, label accuracy, semantics, distributions.
Analyzes all 12 classes (11 themes + Class 0) for data quality before model training.
"""
import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset/ALMA_sentence_level_dataset.csv"

THEMES = ['Attainment', 'First_Gen', 'Aspirational', 'Navigational', 'Resistance',
          'Perseverance', 'Filial_Piety', 'Familial', 'Community_Consciousness',
          'Social', 'Spiritual']

# Theme definitions for keyword/pattern validation
THEME_DEFINITIONS = {
    'Attainment': {
        'desc': 'Concrete achievement of educational/career milestones (degree completion, career placement)',
        'keywords': ['degree', 'graduate', 'diploma', 'career', 'job', 'profession', 'become a',
                     'get my', 'earn', 'achieve', 'accomplish', 'complete', 'finish',
                     'computer science', 'engineering', 'nursing', 'doctor', 'teacher',
                     'bachelor', 'master', 'phd', 'major', 'minor', 'certification'],
        'patterns': [r'get (?:a|my) degree', r'become (?:a|an)', r'want to be', r'career in',
                     r'major(?:ing)? in', r'get into', r'grad(?:uate|uation)']
    },
    'First_Gen': {
        'desc': 'Being the first in family to attend college',
        'keywords': ['first generation', 'first one', 'first person', 'first in my family',
                     'no one in my family', 'nobody in my family', 'first to go',
                     'first to attend', 'parents did not', 'parents never'],
        'patterns': [r'first (?:one|person|generation|in my)', r'no(?:body|one) in my family',
                     r'parents? (?:did not|never|didn)', r'first to (?:go|attend|graduate)']
    },
    'Aspirational': {
        'desc': 'Hopes and dreams for the future, even facing barriers',
        'keywords': ['hope', 'dream', 'aspire', 'goal', 'future', 'want', 'wish',
                     'plan', 'ambition', 'strive', 'passion', 'desire', 'motivated',
                     'looking forward', 'one day', 'someday', 'eventually'],
        'patterns': [r'(?:i |my )(?:want|hope|dream|wish|plan|goal|aspir)',
                     r'in the future', r'one day', r'looking forward', r'i (?:am|was) motivated']
    },
    'Navigational': {
        'desc': 'Skills of maneuvering through institutions, understanding requirements',
        'keywords': ['requirement', 'prerequisite', 'transfer', 'class', 'course', 'major',
                     'credit', 'gpa', 'advisor', 'counselor', 'schedule', 'semester',
                     'curriculum', 'syllabus', 'enroll', 'register', 'apply', 'application',
                     'resource', 'office hours', 'tutoring', 'study', 'learn'],
        'patterns': [r'required? (?:for|class|course)', r'transfer to', r'pre-?req',
                     r'(?:this|the) class', r'(?:this|the) course', r'office hours',
                     r'(?:i|we) (?:need|have) to take', r'help me (?:understand|learn|study)']
    },
    'Resistance': {
        'desc': 'Challenging inequality, overcoming systemic barriers, proving doubters wrong',
        'keywords': ['overcome', 'barrier', 'obstacle', 'challenge', 'fight', 'struggle',
                     'prove', 'despite', 'regardless', 'adversity', 'discrimination',
                     'stereotype', 'doubt', 'underestimate', 'minority', 'underrepresented',
                     'break', 'defy', 'push through', 'persisted'],
        'patterns': [r'prove (?:them|people|everyone) wrong', r'despite (?:the|all)',
                     r'no matter what', r'even (?:though|if|when)', r'break(?:ing)? (?:the|through)',
                     r'i (?:can|will|refuse)', r'(?:over)?com(?:e|ing) (?:the|my|all)']
    },
    'Perseverance': {
        'desc': 'Persistence, grit, determination to continue despite difficulty',
        'keywords': ['keep going', 'push', 'persist', 'persevere', 'determination', 'grit',
                     'never give up', 'don\'t quit', 'stay focused', 'stay strong',
                     'hard work', 'dedication', 'commitment', 'endure', 'tough',
                     'difficult', 'struggle', 'stress', 'anxiety', 'overwhelm',
                     'manage', 'cope', 'deal with', 'get through'],
        'patterns': [r'keep (?:going|pushing|trying|working)', r'never (?:give|gave) up',
                     r'(?:don\'t|do not) (?:quit|stop|give)', r'stay (?:focused|strong|motivated)',
                     r'push(?:ing)? (?:through|myself|forward)', r'(?:work|try|study) hard']
    },
    'Filial_Piety': {
        'desc': 'Respect, duty, and obligation toward parents and elders',
        'keywords': ['parents', 'mother', 'father', 'mom', 'dad', 'family', 'sacrifice',
                     'duty', 'obligation', 'honor', 'respect', 'elder', 'proud',
                     'make them proud', 'for my parents', 'raised me', 'provided'],
        'patterns': [r'(?:make|making) (?:my |them |her |him )proud', r'for my (?:parent|mom|dad|mother|father|family)',
                     r'(?:my |their )sacrifice', r'(?:they|my parents?) (?:sacrificed|worked|raised)',
                     r'(?:honor|respect) (?:my|their)']
    },
    'Familial': {
        'desc': 'Cultural knowledge from familia, community history, family support',
        'keywords': ['family', 'parents', 'mother', 'father', 'brother', 'sister',
                     'sibling', 'grandparent', 'uncle', 'aunt', 'cousin', 'relative',
                     'household', 'home', 'culture', 'heritage', 'tradition', 'upbringing',
                     'support', 'encourage', 'raise'],
        'patterns': [r'my (?:family|parents?|mom|dad|mother|father|brother|sister)',
                     r'(?:family|parents?) (?:support|encourage|help|taught|told)',
                     r'(?:grew|grow) up', r'back home', r'where i(?:\'m| am) from']
    },
    'Community_Consciousness': {
        'desc': 'Awareness of and desire to contribute to community',
        'keywords': ['community', 'give back', 'help others', 'contribute', 'society',
                     'serve', 'impact', 'change', 'difference', 'better world',
                     'volunteer', 'outreach', 'inspire', 'next generation'],
        'patterns': [r'give back', r'help (?:others|people|my community)',
                     r'make (?:a |the )(?:difference|impact|change|world)',
                     r'contribute to', r'(?:my|the) community']
    },
    'Social': {
        'desc': 'Networks of people providing instrumental and emotional support',
        'keywords': ['friend', 'peer', 'classmate', 'study group', 'mentor', 'professor',
                     'teacher', 'tutor', 'connect', 'network', 'relationship', 'bond',
                     'support', 'help each other', 'together', 'team', 'collaborate'],
        'patterns': [r'(?:my |make |new )friend', r'study (?:group|together|with)',
                     r'(?:class|lab)mate', r'(?:my |a )mentor', r'help (?:each other|one another)',
                     r'(?:work|study|learn) together']
    },
    'Spiritual': {
        'desc': 'Spiritual or philosophical sense of purpose and meaning',
        'keywords': ['god', 'faith', 'prayer', 'believe', 'spiritual', 'religion', 'church',
                     'purpose', 'meaning', 'soul', 'blessed', 'grateful', 'thankful',
                     'universe', 'destiny', 'fate', 'calling', 'meant to be',
                     'philosophy', 'existential', 'life purpose', 'higher power'],
        'patterns': [r'(?:i |my )(?:believe|faith|prayer|spirit)', r'(?:god|lord|jesus|allah)',
                     r'meant to be', r'(?:my |a |the )purpose', r'(?:i am |im )(?:grateful|blessed|thankful)',
                     r'higher (?:power|purpose|calling)']
    }
}


def analyze_theme_sentences(df, theme, n_samples=15):
    """Deep analysis of sentences for a single theme."""
    pos = df[df[theme] == 1]
    neg = df[df[theme] == 0]
    defn = THEME_DEFINITIONS[theme]

    print(f"\n{'='*80}")
    print(f"THEME: {theme} ({len(pos)} positive, {len(neg)} negative)")
    print(f"Definition: {defn['desc']}")
    print(f"{'='*80}")

    # ---- 1. Keyword coverage analysis ----
    keywords = defn['keywords']
    patterns = defn['patterns']

    pos_texts = pos['sentence'].str.lower()
    neg_texts = neg['sentence'].str.lower()

    # How many positive sentences contain at least 1 keyword?
    kw_match_pos = 0
    kw_match_neg = 0
    pattern_match_pos = 0
    pattern_match_neg = 0

    for _, row in pos.iterrows():
        sent = str(row['sentence']).lower()
        if any(kw in sent for kw in keywords):
            kw_match_pos += 1
        if any(re.search(p, sent) for p in patterns):
            pattern_match_pos += 1

    # Sample negative for efficiency
    neg_sample = neg.sample(min(2000, len(neg)), random_state=42)
    for _, row in neg_sample.iterrows():
        sent = str(row['sentence']).lower()
        if any(kw in sent for kw in keywords):
            kw_match_neg += 1
        if any(re.search(p, sent) for p in patterns):
            pattern_match_neg += 1

    kw_pos_pct = kw_match_pos / len(pos) * 100 if len(pos) > 0 else 0
    kw_neg_pct = kw_match_neg / len(neg_sample) * 100 if len(neg_sample) > 0 else 0
    pat_pos_pct = pattern_match_pos / len(pos) * 100 if len(pos) > 0 else 0
    pat_neg_pct = pattern_match_neg / len(neg_sample) * 100 if len(neg_sample) > 0 else 0

    print(f"\n  KEYWORD VALIDATION:")
    print(f"    Positive sentences with keywords: {kw_match_pos}/{len(pos)} ({kw_pos_pct:.1f}%)")
    print(f"    Negative sentences with keywords: {kw_match_neg}/{len(neg_sample)} ({kw_neg_pct:.1f}%) [sampled]")
    print(f"    Pattern match (positive): {pattern_match_pos}/{len(pos)} ({pat_pos_pct:.1f}%)")
    print(f"    Pattern match (negative): {pattern_match_neg}/{len(neg_sample)} ({pat_neg_pct:.1f}%) [sampled]")

    # ---- 2. Potential mislabels ----
    # Positive sentences WITHOUT any keyword match (potential false positives)
    print(f"\n  POTENTIAL FALSE POSITIVES (positive but no keyword match):")
    no_kw_pos = []
    for _, row in pos.iterrows():
        sent = str(row['sentence']).lower()
        if not any(kw in sent for kw in keywords):
            no_kw_pos.append(row)
    if no_kw_pos:
        fp_sample = no_kw_pos[:min(5, len(no_kw_pos))]
        for r in fp_sample:
            print(f"    - [{r['alma_id']}] \"{str(r['sentence'])[:120]}\"")
        print(f"    ... {len(no_kw_pos)} total ({len(no_kw_pos)/len(pos)*100:.1f}% of positive)")
    else:
        print(f"    None found - all positive sentences contain relevant keywords")

    # Negative sentences WITH keyword match (potential false negatives)
    print(f"\n  POTENTIAL FALSE NEGATIVES (negative but HAS keywords):")
    has_kw_neg = []
    for _, row in neg_sample.iterrows():
        sent = str(row['sentence']).lower()
        if any(kw in sent for kw in keywords):
            has_kw_neg.append(row)
    if has_kw_neg:
        fn_sample = has_kw_neg[:min(5, len(has_kw_neg))]
        for r in fn_sample:
            matched_kw = [kw for kw in keywords if kw in str(r['sentence']).lower()]
            print(f"    - [{r['alma_id']}] \"{str(r['sentence'])[:100]}\" → kw: {matched_kw[:3]}")
        print(f"    ... {len(has_kw_neg)} total ({len(has_kw_neg)/len(neg_sample)*100:.1f}% of sampled neg)")
    else:
        print(f"    None found")

    # ---- 3. Top words unique to this theme ----
    print(f"\n  TOP DISCRIMINATIVE WORDS (frequent in positive, rare in negative):")
    stopwords = {'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'am', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'that',
                 'this', 'it', 'and', 'or', 'but', 'not', 'so', 'as', 'if', 'when', 'than',
                 'do', 'did', 'have', 'has', 'had', 'will', 'would', 'can', 'could',
                 'should', 'may', 'might', 'just', 'also', 'very', 'really', 'about',
                 'more', 'out', 'up', 'all', 'there', 'what', 'which', 'who', 'how',
                 'here', 'because', 'its', 'they', 'them', 'their', 'he', 'she', 'her',
                 'his', 'we', 'our', 'you', 'your', 'no', 'like', 'get', 'got', 'know',
                 'think', 'make', 'go', 'going', 'one', 'even', 'being', 'well', 'still',
                 'into', 'some', 'only', 'much', 'through', 'after', 'then', 'other',
                 'new', 'now', 'way', 'many', 'these', 'been', 'thing', 'things'}

    pos_words = Counter()
    neg_words = Counter()
    for sent in pos['sentence'].values:
        words = re.findall(r'[a-z]+', str(sent).lower())
        pos_words.update(w for w in words if w not in stopwords and len(w) > 2)
    for sent in neg_sample['sentence'].values:
        words = re.findall(r'[a-z]+', str(sent).lower())
        neg_words.update(w for w in words if w not in stopwords and len(w) > 2)

    # Compute discriminative score
    total_pos = sum(pos_words.values())
    total_neg = sum(neg_words.values())
    disc_scores = {}
    for word, count in pos_words.items():
        if count >= 5:
            pos_rate = count / total_pos
            neg_rate = (neg_words.get(word, 0) + 1) / total_neg  # +1 smoothing
            disc_scores[word] = pos_rate / neg_rate

    top_disc = sorted(disc_scores.items(), key=lambda x: -x[1])[:15]
    for word, score in top_disc:
        print(f"    {word:<20} score={score:.1f}  (pos={pos_words[word]}, neg={neg_words.get(word,0)})")

    # ---- 4. Sample positive sentences ----
    print(f"\n  SAMPLE POSITIVE SENTENCES ({min(n_samples, len(pos))} of {len(pos)}):")
    samples = pos.sample(min(n_samples, len(pos)), random_state=42)
    for _, r in samples.iterrows():
        other_themes = [t for t in THEMES if t != theme and r[t] == 1]
        multi = f" [+{','.join(other_themes)}]" if other_themes else ""
        print(f"    [{r['alma_id']}] \"{str(r['sentence'])[:140]}\"{multi}")

    # ---- 5. Sentence length distribution ----
    print(f"\n  LENGTH STATS:")
    print(f"    Positive: mean={pos['sentence_length'].mean():.0f}, median={pos['sentence_length'].median():.0f}, "
          f"std={pos['sentence_length'].std():.0f}")
    print(f"    Negative: mean={neg['sentence_length'].mean():.0f}, median={neg['sentence_length'].median():.0f}, "
          f"std={neg['sentence_length'].std():.0f}")

    return {
        'theme': theme,
        'pos_count': len(pos),
        'neg_count': len(neg),
        'kw_coverage_pos': kw_pos_pct,
        'kw_coverage_neg': kw_neg_pct,
        'potential_fp': len(no_kw_pos),
        'potential_fn': len(has_kw_neg),
        'fp_rate': len(no_kw_pos) / len(pos) * 100 if len(pos) > 0 else 0,
    }


def analyze_class0(df):
    """Analyze Class 0 (no theme) sentences."""
    class0 = df[df[THEMES].sum(axis=1) == 0]

    print(f"\n{'='*80}")
    print(f"CLASS 0: NO THEME ({len(class0)} sentences, {len(class0)/len(df)*100:.1f}% of total)")
    print(f"{'='*80}")

    # Length distribution
    print(f"\n  LENGTH STATS:")
    print(f"    Mean: {class0['sentence_length'].mean():.0f} chars")
    print(f"    Median: {class0['sentence_length'].median():.0f} chars")
    print(f"    Very short (<30 chars): {(class0['sentence_length'] < 30).sum()} ({(class0['sentence_length'] < 30).sum()/len(class0)*100:.1f}%)")
    print(f"    Very long (>300 chars): {(class0['sentence_length'] > 300).sum()} ({(class0['sentence_length'] > 300).sum()/len(class0)*100:.1f}%)")

    # Top words in Class 0
    print(f"\n  TOP WORDS IN CLASS 0:")
    stopwords = {'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'am', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'that',
                 'this', 'it', 'and', 'or', 'but', 'not', 'so', 'as', 'if', 'when', 'than',
                 'do', 'did', 'have', 'has', 'had', 'will', 'would', 'can', 'could',
                 'should', 'may', 'might', 'just', 'also', 'very', 'really', 'about',
                 'more', 'out', 'up', 'all', 'there', 'what', 'which', 'who', 'how',
                 'here', 'because', 'its', 'they', 'them', 'their', 'he', 'she', 'her',
                 'his', 'we', 'our', 'you', 'your', 'no', 'like', 'get', 'got', 'know',
                 'think', 'make', 'go', 'going', 'one', 'even', 'being', 'well', 'still',
                 'into', 'some', 'only', 'much', 'through', 'after', 'then', 'other',
                 'new', 'now', 'way', 'many', 'these', 'been', 'thing', 'things'}
    c0_words = Counter()
    for sent in class0['sentence'].values:
        words = re.findall(r'[a-z]+', str(sent).lower())
        c0_words.update(w for w in words if w not in stopwords and len(w) > 2)
    for word, count in c0_words.most_common(20):
        print(f"    {word:<20} {count}")

    # Check for sentences that LOOK like they should have a theme
    print(f"\n  POTENTIAL MISLABELED CLASS 0 (sentences with strong theme keywords):")
    suspicious = []
    all_keywords = {}
    for theme, defn in THEME_DEFINITIONS.items():
        all_keywords[theme] = defn['keywords']

    c0_sample = class0.sample(min(3000, len(class0)), random_state=42)
    for _, row in c0_sample.iterrows():
        sent = str(row['sentence']).lower()
        matched_themes = []
        for theme, kws in all_keywords.items():
            strong_kws = [kw for kw in kws if kw in sent and len(kw) > 5]
            if len(strong_kws) >= 2:
                matched_themes.append((theme, strong_kws))
        if matched_themes:
            suspicious.append((row, matched_themes))

    if suspicious:
        for (r, themes), _ in zip(suspicious[:10], range(10)):
            theme_str = "; ".join(f"{t}: {kws}" for t, kws in themes)
            print(f"    [{r['alma_id']}] \"{str(r['sentence'])[:120]}\"")
            print(f"      → matches: {theme_str}")
        print(f"    ... {len(suspicious)} total suspicious Class 0 sentences out of {len(c0_sample)} sampled")
    else:
        print(f"    None found")

    # Sample Class 0 sentences
    print(f"\n  SAMPLE CLASS 0 SENTENCES (20 random):")
    samples = class0.sample(min(20, len(class0)), random_state=42)
    for _, r in samples.iterrows():
        print(f"    [{r['alma_id']}] \"{str(r['sentence'])[:140]}\"")

    # By prompt
    print(f"\n  CLASS 0 BY PROMPT:")
    for prompt, grp in class0.groupby('prompt'):
        total_prompt = len(df[df['prompt'] == prompt])
        print(f"    {prompt}: {len(grp)} class0 / {total_prompt} total ({len(grp)/total_prompt*100:.1f}%)")


def cross_theme_analysis(df):
    """Analyze theme co-occurrence and overlap patterns."""
    print(f"\n{'='*80}")
    print(f"CROSS-THEME CO-OCCURRENCE ANALYSIS")
    print(f"{'='*80}")

    # Co-occurrence matrix
    print(f"\n  CO-OCCURRENCE MATRIX (# sentences with BOTH themes):")
    print(f"  {'':>20}", end='')
    for t in THEMES:
        print(f" {t[:5]:>6}", end='')
    print()

    for t1 in THEMES:
        print(f"  {t1:>20}", end='')
        for t2 in THEMES:
            both = ((df[t1] == 1) & (df[t2] == 1)).sum()
            print(f" {both:>6}", end='')
        print()

    # Jaccard similarity
    print(f"\n  TOP THEME PAIRS BY JACCARD SIMILARITY:")
    pairs = []
    for i, t1 in enumerate(THEMES):
        for t2 in THEMES[i+1:]:
            both = ((df[t1] == 1) & (df[t2] == 1)).sum()
            either = ((df[t1] == 1) | (df[t2] == 1)).sum()
            jaccard = both / either if either > 0 else 0
            pairs.append((t1, t2, jaccard, both))
    pairs.sort(key=lambda x: -x[2])
    for t1, t2, j, both in pairs[:10]:
        print(f"    {t1} + {t2}: Jaccard={j:.3f} ({both} co-occurring)")

    # Exclusive themes (only this theme, no others)
    print(f"\n  THEME EXCLUSIVITY (sentences with ONLY this theme):")
    for theme in THEMES:
        only_this = ((df[theme] == 1) & (df[THEMES].sum(axis=1) == 1)).sum()
        total_pos = (df[theme] == 1).sum()
        pct = only_this / total_pos * 100 if total_pos > 0 else 0
        print(f"    {theme:<25} {only_this:>5} exclusive / {total_pos:>5} total ({pct:.1f}%)")


def embedding_similarity_analysis(df):
    """Analyze semantic similarity within and across themes using simple TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"\n{'='*80}")
    print(f"SEMANTIC SIMILARITY ANALYSIS (TF-IDF)")
    print(f"{'='*80}")

    # Get centroid of each theme's sentences
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)
    all_sents = df['sentence'].fillna('').values
    tfidf_matrix = vectorizer.fit_transform(all_sents)

    # Compute theme centroids
    centroids = {}
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            centroids[theme] = np.asarray(tfidf_matrix[mask].mean(axis=0))

    # Class 0 centroid
    class0_mask = df[THEMES].sum(axis=1) == 0
    centroids['Class_0'] = np.asarray(tfidf_matrix[class0_mask].mean(axis=0))

    # Pairwise centroid similarity
    labels = list(centroids.keys())
    n = len(labels)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = cosine_similarity(centroids[labels[i]], centroids[labels[j]])[0][0]

    print(f"\n  CENTROID SIMILARITY MATRIX:")
    print(f"  {'':>20}", end='')
    for l in labels:
        print(f" {l[:6]:>7}", end='')
    print()
    for i, l1 in enumerate(labels):
        print(f"  {l1:>20}", end='')
        for j in range(len(labels)):
            print(f" {sim_matrix[i][j]:>7.3f}", end='')
        print()

    # Most similar and most different theme pairs
    print(f"\n  THEME PAIRS MOST SEMANTICALLY SIMILAR:")
    pairs = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            pairs.append((labels[i], labels[j], sim_matrix[i][j]))
    pairs.sort(key=lambda x: -x[2])
    for l1, l2, sim in pairs[:8]:
        print(f"    {l1} ↔ {l2}: {sim:.3f}")

    print(f"\n  THEME PAIRS MOST SEMANTICALLY DISTINCT:")
    for l1, l2, sim in pairs[-5:]:
        print(f"    {l1} ↔ {l2}: {sim:.3f}")

    # Intra-theme coherence (avg similarity within a theme's sentences)
    print(f"\n  INTRA-THEME COHERENCE (avg pairwise similarity within theme):")
    for theme in THEMES + ['Class_0']:
        if theme == 'Class_0':
            mask = df[THEMES].sum(axis=1) == 0
        else:
            mask = df[theme] == 1
        if mask.sum() < 10:
            continue
        # Sample for efficiency
        indices = np.where(mask)[0]
        sample_idx = np.random.choice(indices, min(200, len(indices)), replace=False)
        sub_matrix = tfidf_matrix[sample_idx].toarray()
        sims = cosine_similarity(sub_matrix)
        # Average off-diagonal
        np.fill_diagonal(sims, 0)
        avg_sim = sims.sum() / (len(sims) * (len(sims) - 1))
        print(f"    {theme:<25} coherence={avg_sim:.4f} (n={mask.sum()})")

    # Top TF-IDF terms per theme
    print(f"\n  TOP TF-IDF TERMS PER THEME:")
    feature_names = vectorizer.get_feature_names_out()
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() == 0:
            continue
        mean_tfidf = np.array(tfidf_matrix[mask].mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-10:][::-1]
        top_terms = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
        terms_str = ", ".join(f"{t}({s:.3f})" for t, s in top_terms)
        print(f"    {theme:<25}: {terms_str}")


def overall_quality_summary(results):
    """Print overall data quality summary."""
    print(f"\n{'='*80}")
    print(f"OVERALL DATA QUALITY SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Theme':<25} {'Count':>6} {'KW%':>6} {'FP%':>6} {'Quality':>10}")
    print(f"  {'-'*60}")
    for r in results:
        quality = 'GOOD' if r['kw_coverage_pos'] >= 60 and r['fp_rate'] < 40 else \
                  'FAIR' if r['kw_coverage_pos'] >= 40 else 'CHECK'
        print(f"  {r['theme']:<25} {r['pos_count']:>6} {r['kw_coverage_pos']:>5.1f}% {r['fp_rate']:>5.1f}% {quality:>10}")

    print(f"\n  NOTES:")
    print(f"  - KW%: % of positive sentences containing theme keywords (higher = more consistent)")
    print(f"  - FP%: % of positive sentences with NO keyword match (potential false positives)")
    print(f"  - Low KW% may indicate: (1) keywords list too narrow, (2) subtle theme expression,")
    print(f"    or (3) genuine mislabels. Manual review of flagged sentences recommended.")
    print(f"  - Themes like Perseverance, Navigational express broadly → lower keyword precision expected.")


def main():
    print("Loading sentence-level dataset...")
    df = pd.read_csv(DATA)
    print(f"Loaded: {len(df)} sentences, {df['essay_id'].nunique()} essays\n")

    # ---- Per-theme analysis ----
    results = []
    for theme in THEMES:
        r = analyze_theme_sentences(df, theme)
        results.append(r)

    # ---- Class 0 analysis ----
    analyze_class0(df)

    # ---- Cross-theme analysis ----
    cross_theme_analysis(df)

    # ---- Semantic analysis ----
    try:
        embedding_similarity_analysis(df)
    except ImportError:
        print("\n[SKIPPING] sklearn not available for TF-IDF analysis")

    # ---- Quality summary ----
    overall_quality_summary(results)


if __name__ == '__main__':
    main()
