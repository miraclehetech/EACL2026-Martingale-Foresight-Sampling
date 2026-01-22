import json
import re
from collections import Counter

def most_common_element(lst):
    """Return the most common element in a list."""
    counts = Counter(lst)
    max_count = max(counts.values())
    most_common = [k for k, v in counts.items() if v == max_count][0]
    return most_common

def extract_numbers(input_string):
    """Extract all numbers from string."""
    return re.findall(r'\d+', input_string)

def find_last_number(input_string):
    """Find the last number in a string."""
    words = input_string.split()
    numbers = []
    for word in words:
        try:
            number = float(word)
            numbers.append(number)
        except ValueError:
            pass

    if not numbers:
        return ""
    return str(numbers[-1])

def eval_cot_answer(pred, gt):
    """
    Evaluate chain-of-thought answer against ground truth.
    Returns (is_correct, extracted_answer).
    """
    boxed_contents = ""
    try:
        if "\\boxed{" in pred:
            boxed_contents = re.findall(r'\\boxed\{(.*?)\}', pred)
            if boxed_contents:
                boxed_contents = boxed_contents[-1]
            else:
                boxed_contents = ""
        elif "<boxed>" in pred:
            boxed_contents = re.findall(r'<boxed>(\d+)<\/boxed>', pred)
            if boxed_contents:
                boxed_contents = boxed_contents[-1].strip()
            else:
                boxed_contents = ""
        elif "The answer is:" in pred:
            boxed_contents = pred.split("The answer is:")[-1].strip()
        else:
            # Fallback: parse from last 50 characters
            boxed_contents = find_last_number(pred[-50:])
    except:
        return False, None
    
    answer = boxed_contents.strip('\\').replace(",", "").strip("$").strip()
    
    # Handle different answer formats
    if "." in answer:
        pred_ans = answer
    elif "frac" in answer and len(extract_numbers(answer)) == 2:
        if float(extract_numbers(answer)[1]) != 0:
            pred_ans = float(extract_numbers(answer)[0]) / float(extract_numbers(answer)[1])
        else:
            return False, None
    elif extract_numbers(answer):
        pred_ans = extract_numbers(answer)[0]
    else:
        return False, None

    # Compare with ground truth
    try:
        if abs(float(gt) - float(pred_ans)) < 1e-3:
            return True, pred_ans
        else:
            return False, pred_ans
    except:
        return False, None

# Load predictions
prediction = []
with open(f"/mnt/d/phi-decoding/results/mfs_test_ours_gsm.json") as file:
    for line in file:
        prediction.append(json.loads(line))

print(len(prediction))

# Evaluate predictions
correct_num = 0
correct_num_passk = 0
correct_num_sc = 0

for i in range(len(prediction)):
    gt = prediction[i]['ground_truth']

    # Check if main response is correct
    if eval_cot_answer(prediction[i]['response'], gt)[0]:
        correct_num += 1

    # Calculate Pass@K and Self-Consistency@K
    correct_passk = False
    candidate_list = []
    
    if "response_all_beams" in prediction[i]:
        for j in range(len(prediction[i]['response_all_beams'])):
            # Calculate Pass@K
            if eval_cot_answer(prediction[i]['response_all_beams'][j], gt)[0]:
                correct_passk = True

            # Collect answers for Self-Consistency@K
            if eval_cot_answer(prediction[i]['response_all_beams'][j], gt)[1]:
                candidate_list.append(float(eval_cot_answer(prediction[i]['response_all_beams'][j], gt)[1]))

        if correct_passk:
            correct_num_passk += 1

        # Self-Consistency: majority voting
        try:
            pred = most_common_element(candidate_list)
            if abs(float(gt) - float(pred)) < 1e-5:
                correct_num_sc += 1
        except:
            continue

# Print results
print("Acc: ", correct_num / len(prediction))
print("Pass@K: ", correct_num_passk / len(prediction))
print("SC@K: ", correct_num_sc / len(prediction))
