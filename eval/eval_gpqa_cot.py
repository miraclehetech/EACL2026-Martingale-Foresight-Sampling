import json
from collections import Counter

def getAnswer(response):
    """Extract uppercase letter answer (A-E) from response."""
    pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char
    return ""

def is_empty(ans):
    """Check if answer is empty or None."""
    return ans is None or (isinstance(ans, str) and ans.strip() == "")

# Load predictions
prediction = []
with open(f"/mnt/d/phi-decoding/result/ours-mistral-gpqa-test.json") as file:
    for line in file:
        prediction.append(json.loads(line))

# Load ground truth
with open(f"/mnt/d/phi-Decoding/data/gpqa_main_test.json") as file:
    ground_truth = json.load(file)

print(len(prediction))

# Evaluate predictions using majority voting
correct_num = 0
deduct = 0
output = []

for i in range(len(prediction)):
    response = prediction[i]['all_responses']
    answers = [getAnswer(a) for a in response if a]
    
    # Filter out empty answers
    valid_answers = [a.strip() if isinstance(a, str) else a
                     for a in answers if not is_empty(a)]
    
    # Skip if all answers are empty
    if len(valid_answers) == 0:
        print(answers)
        deduct += 1
        continue
    
    # Get majority vote
    pred = Counter(valid_answers).most_common(1)[0][0]
    gt = ground_truth[i]['target']
    
    try:
        if gt == pred:
            correct_num += 1
        else:
            output.append({
                "id": i,
                "ground_truth": gt,
                "pred": pred
            })
    except:
        continue

# Save incorrect predictions
with open("/mnt/d/phi-Decoding/result/ours-gpqa-test-wrong.json", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")

# Print results
print(correct_num)
print(len(prediction) - deduct)
print(correct_num / (len(prediction) - deduct))
