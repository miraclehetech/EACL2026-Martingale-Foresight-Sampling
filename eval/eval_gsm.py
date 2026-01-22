import json
import re
from collections import Counter

def normalize_number(s: str):
    """Normalize numerical string to pure number string."""
    if s is None:
        return None
    s = s.strip()

    # Extract fractions (a/b format)
    frac = re.search(r'([+-]?\d+)\s*/\s*(\d+)', s)
    if frac and not re.search(r'\d', s.replace(frac.group(0), '')):
        num = int(frac.group(1))
        den = int(frac.group(2))
        return f"{num}/{den}"

    # Extract numbers with optional $, signs, thousands separators, decimals, and percentages
    m = re.findall(
        r'[+-]?\s*\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%?',
        s
    )
    if not m:
        return None
    
    cand = m[-1]  # Take the last match
    cand = cand.replace(' ', '')

    # Check for percentage
    is_percent = cand.endswith('%')
    if is_percent:
        cand = cand[:-1]

    # Remove currency symbols and thousand separators
    cand = cand.replace('$', '').replace(',', '')

    # Edge case: only sign or empty
    if cand in ('', '+', '-'):
        return None

    return cand

def extract_answer(text: str):
    """Extract answer from text, supporting various formats."""
    if not text:
        return None

    # Check for LaTeX \boxed{...}
    boxed = re.findall(r'\\boxed\{(.*?)\}', text, flags=re.DOTALL)
    if boxed:
        return normalize_number(boxed[-1])

    # Check for <boxed>...</boxed>
    boxed2 = re.findall(r'<boxed>(.*?)<\/boxed>', text, flags=re.DOTALL|re.IGNORECASE)
    if boxed2:
        return normalize_number(boxed2[-1])

    # Fallback: extract number from plain text
    return normalize_number(text)

def safe_float(x):
    """Safely convert to float, handling fraction format."""
    try:
        if isinstance(x, str) and '/' in x and x.count('/') == 1:
            a, b = x.split('/')
            return float(a) / float(b)
        return float(x)
    except:
        return None

# File paths
input_path = "/mnt/d/phi-Decoding/result/ours-llama31-gsm-test-08.json"
wrong_out = "/mnt/d/phi-Decoding/result/ours-llama31-gsm-test-wrong.json"

# Load predictions and ground truth
with open(input_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

ground_truth = json.load(open("/mnt/d/phi-Decoding/data/gsm_test.json"))
ground_truth = [item['target'] for item in ground_truth]

# Evaluate predictions
output = []
correct = 0

for i, item in enumerate(samples):
    beams = item.get("all_responses", []) or []
    answers = [extract_answer(b) for b in beams if b]
    answers = [a for a in answers if a is not None]

    # Get majority vote
    vote = Counter(answers)
    majority = vote.most_common(1)[0][0] if vote else None

    gt_raw = str(ground_truth[i])
    gt = normalize_number(gt_raw)

    # Check correctness using floating point comparison
    p_f = safe_float(majority)
    g_f = safe_float(gt)

    if p_f is not None and g_f is not None:
        is_correct = abs(p_f - g_f) < 1e-5
    else:
        is_correct = (majority is not None and gt is not None and majority == gt)

    item["majority_answer"] = majority
    item["is_correct"] = is_correct

    if is_correct:
        correct += 1
    else:
        output.append({
            "id": item.get("id"),
            "ground_truth": gt_raw,
            "pred": majority
        })

# Save incorrect predictions
with open(wrong_out, "w", encoding="utf-8") as f:
    for it in output:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

# Print accuracy
print(f"Accuracy: {correct}/{len(samples)} = {correct/len(samples):.2%}")
