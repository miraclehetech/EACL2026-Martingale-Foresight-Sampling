import json
from collections import Counter

def getAnswer(response):
    """Extract uppercase letter answer (A-D) from response."""
    pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D"]:
            return char
    return ""

# Mapping from letter to index
map_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

# Load predictions
prediction = []
idx = 0
with open(f"/mnt/d/phi-Decoding/result/ours-mistral-logiqa-test-1.0.json") as file:
    for line in file:
        idx += 1
        try:
            prediction.append(json.loads(line))
        except:
            print(idx, ' error')

# Load ground truth
gt = json.load(open("/mnt/d/phi-Decoding/data/logiqa_test.json"))

# Evaluate predictions using majority voting
output = []
correct_num = 0
deduct = 0

for i in range(len(prediction)):
    answers = [getAnswer(a) for a in prediction[i]['all_responses'] if a]
    judge = Counter(answers).most_common(1)[0][0]
    
    if judge == "":
        deduct += 1
        continue
    
    majority = Counter([a for a in answers if a != ""]).most_common(1)[0][0]
    pred = map_dict[majority]
    
    ground_truth = gt[i]['label']
    if ground_truth == pred:
        correct_num += 1
    else:
        output.append({
            "id": i,
            "ground_truth": ground_truth,
            "pred": pred
        })

# Save incorrect predictions
with open("/mnt/d/phi-Decoding/result/ours-logiqa-test-wrong.json", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")

# Print results
print(correct_num)
print(len(prediction))
print(correct_num / (len(prediction) - deduct))
