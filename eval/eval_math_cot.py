import json
import re
from collections import Counter

# Regular expression for signed numbers
SIGNED_NUMBER_RE = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

def most_common_element(lst):
    """Return the most common element in a list."""
    counts = Counter(lst)
    max_count = max(counts.values())
    most_common = [k for k, v in counts.items() if v == max_count][0]
    return most_common

def extract_numbers(input_string):
    """Return all signed numbers (including decimals and scientific notation) as string list."""
    return re.findall(SIGNED_NUMBER_RE, input_string)

def find_last_number(input_string):
    """Return the last signed number (empty string if none found)."""
    nums = re.findall(SIGNED_NUMBER_RE, input_string)
    return nums[-1] if nums else ""

def _strip_tex_wrappers(s: str) -> str:
    """Remove common LaTeX wrappers and formatting."""
    s = s.replace(",", "")
    s = s.strip()
    s = s.strip("$")
    s = s.replace(r"\left", "").replace(r"\right", "")
    return s

def _parse_latex_fraction(ans: str):
    """
    Parse answers containing \\frac, supporting various formats.
    Returns float on success, None otherwise.
    """
    s = _strip_tex_wrappers(ans)
    
    # Check for leading negative sign: -\frac{...}{...}
    lead_neg = bool(re.search(r'(^|[^\d])-\s*\\frac', s))
    
    # Extract \frac{num}{den}
    m = re.search(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', s)
    if not m:
        return None
    num_s, den_s = m.group(1), m.group(2)

    # Extract one signed number from numerator and denominator
    num_list = re.findall(SIGNED_NUMBER_RE, num_s)
    den_list = re.findall(SIGNED_NUMBER_RE, den_s)
    if not num_list or not den_list:
        return None

    num = float(num_list[0])
    den = float(den_list[0])
    if den == 0:
        return None

    val = num / den
    if lead_neg:
        val = -val
    return val

def eval_cot_answer(pred, gt):
    """
    Evaluate chain-of-thought answer against ground truth.
    Returns (is_correct, extracted_answer).
    """
    boxed_contents = ""
    try:
        # Check for \boxed{...}
        if "\\boxed{" in pred:
            boxed_list = re.findall(r'\\boxed\{(.*?)\}', pred, flags=re.DOTALL)
            boxed_contents = boxed_list[-1] if boxed_list else ""
        # Check for <boxed>...</boxed>
        elif "<boxed>" in pred:
            boxed_list = re.findall(r'<boxed>(.*?)</boxed>', pred, flags=re.DOTALL)
            boxed_contents = boxed_list[-1].strip() if boxed_list else ""
        # Check for "answer is:"
        elif "answer is:" in pred:
            boxed_contents = pred.split("answer is:")[-1].strip()
        else:
            # Fallback: find last signed number in the ending segment
            boxed_contents = pred[-100:]
    except:
        return False, None

    answer = _strip_tex_wrappers(boxed_contents)

    # Handle \frac format first
    if "frac" in answer:
        frac_val = _parse_latex_fraction(answer)
        if frac_val is not None:
            pred_ans = frac_val
        else:
            # Try regular "a/b" format
            m = re.search(rf'({SIGNED_NUMBER_RE})\s*/\s*({SIGNED_NUMBER_RE})', answer)
            if m:
                num = float(m.group(1))
                den = float(m.group(2))
                if den == 0:
                    return False, None
                pred_ans = num / den
            else:
                # Fallback: find last signed number
                last_num = find_last_number(answer)
                if last_num == "":
                    return False, None
                pred_ans = last_num
    else:
        # Non-\frac case: directly find last signed number
        last_num = find_last_number(answer)
        if last_num == "":
            return False, None
        pred_ans = last_num

    # Compare with ground truth
    try:
        if abs(float(gt) - float(pred_ans)) < 1e-5:
            return True, pred_ans
        else:
            return False, pred_ans
    except:
        return False, None

# Load predictions and ground truth
prediction = []
with open(f"../result/8_8_10_math-test.json") as file:
    for line in file:
        prediction.append(json.loads(line))

print(len(prediction))

with open(f"/mnt/d/phi-Decoding/data/math-500.json") as file:
    ground_truth = json.load(file)

# Evaluate predictions
correct_num = 0
correct_num_passk = 0
correct_num_sc = 0
output = []

for i in range(len(prediction)):
    # Check if main response is correct
    if eval_cot_answer(prediction[i]['response'], ground_truth[i]['target'])[0]:
        correct_num += 1

    # Calculate Pass@K and Self-Consistency@K
    correct_passk = False
    candidate_list = []
    
    if "all_responses" in prediction[i]:
        for j in range(len(prediction[i]['all_responses'])):
            # Calculate Pass@K
            if eval_cot_answer(prediction[i]['all_responses'][j], ground_truth[i]['target'])[0]:
                correct_passk = True

            # Collect answers for Self-Consistency@K
            if eval_cot_answer(prediction[i]['all_responses'][j], ground_truth[i]['target'])[1]:
                candidate_list.append(float(eval_cot_answer(prediction[i]['all_responses'][j], ground_truth[i]['target'])[1]))

        if correct_passk:
            correct_num_passk += 1

        # Self-Consistency: majority voting
        try:
            pred = most_common_element(candidate_list)
            if abs(float(ground_truth[i]['target']) - float(pred)) < 1e-5:
                correct_num_sc += 1
            else:
                output.append({
                    "id": i,
                    "ground_truth": ground_truth[i]['target'],
                    "pred": pred
                })
        except:
            continue

# Save incorrect predictions
with open("/mnt/d/phi-Decoding/result/ours-math-test-wrong.json", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")

# Print results
print("Acc: ", correct_num / len(prediction))
print("Pass@K: ", correct_num_passk / len(prediction))
print("SC@K: ", correct_num_sc / len(prediction))
