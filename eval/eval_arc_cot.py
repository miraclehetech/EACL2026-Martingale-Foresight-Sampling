import json
from collections import Counter

def getAnswer(response):
    """Extract uppercase letter answer (A, B, C, D) from response."""
    pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D"]:
            return char
    return ""

def getAnswer2(response):
    """Extract numeric answer (1, 2, 3, 4) from response."""
    pred = response.split("answer is")[-1]
    for char in pred:
        if char in ["1", "2", "3", "4"]:
            return char
    return ""

# Load predictions
prediction = []
idx = 0
with open(f"/mnt/d/phi-Decoding/results/ours-qwen25-arc-c-test-08.json") as file:
    for line in file:
        idx += 1
        try:
            prediction.append(json.loads(line))
        except:
            print(idx, ' error')

# Load ground truth
gt = json.load(open("/mnt/d/phi-Decoding/data/arc-c_test.json"))
gt = [item['target'] for item in gt]

# Evaluate predictions
output = []
correct_num = 0
deduct = 0

for i in range(len(prediction)):
    # Handle numeric or letter answers based on ground truth format
    if gt[i] in ["1", "2", "3", "4"]:
        answers = [getAnswer2(a) for a in prediction[i]['all_responses'] if a]
    else:
        answers = [getAnswer(a) for a in prediction[i]['all_responses'] if a]
    
    judge = Counter(answers).most_common(1)[0][0]
    if judge == "":
        deduct += 1
        continue
    
    # Get majority vote
    majority = Counter([a for a in answers if a != ""]).most_common(1)[0][0]
    pred = majority
    
    ground_truth = gt[i]
    if ground_truth == pred:
        correct_num += 1
    else:
        output.append({
            "id": i,
            "ground_truth": ground_truth,
            "pred": pred
        })

# Save incorrect predictions
with open("/mnt/d/phi-Decoding/result/ours-arc-c-test-mistral-according-to-phi-wrong.json", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")

# Print results
print(correct_num)
print(len(prediction) - deduct)
print(correct_num / (len(prediction) - deduct))
