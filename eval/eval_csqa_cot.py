import json

def getAnswer(response):
    """Extract uppercase letter answer (A-E) from response."""
    pred = response.split("The answer is:")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char
    return ""

# Load predictions
prediction = []
with open(f"/cpfs01/user/xufangzhi/o1/cluster_results/241228-14.json") as file:
    for line in file:
        prediction.append(json.loads(line))

print(len(prediction))

# Evaluate predictions
correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response']
    pred = getAnswer(response)
    gt = prediction[i]['ground_truth']
    
    try:
        if gt == pred:
            correct_num += 1
    except:
        continue

# Print accuracy
print(correct_num / len(prediction))
