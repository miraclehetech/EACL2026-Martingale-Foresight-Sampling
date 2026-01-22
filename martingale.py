import json
import os
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import copy
from huggingface_hub import login

# Configuration
dataset = 'gsm'
dataset_path = './data/gsm_test.json'
seed = 42

# Model configuration
model_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Output paths
output_path = '/mnt/d/phi-Decoding/result/ours-llama31-gsm-test-08.json'
output_path2 = '/mnt/d/phi-Decoding/result/ours-llama31-gsm-test-record-08.json'

# Algorithm parameters
step_beam_size = 8
num_foresight = 8
num_rollout = 8
sigma_rate = 0.8
minimum_foresight_steps = 0
EARLY_STOPPING_ADVANTAGE = 0.0001

# Login to HuggingFace (replace with your token or use environment variable)
# login("your_hf_token_here")
login()

def prepare_chat_template(example, system_prompt):
    """Prepare chat template for the model."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"The question: {example['input']}\nPlease directly output the reasoning steps.\n"},
    ]


# Load dataset
with open(dataset_path) as f:
    test_data = json.load(f)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, max_length=32768)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
model = LLM(model=model_path, tensor_parallel_size=2, trust_remote_code=True, max_model_len=32768)
np.random.seed(seed)


def process_step(example, system_prompt, active_beams):
    """
    Process a single reasoning step in the beam search algorithm.
    Generates candidate paths and evaluates them using foresight.
    """
    # Step 1: Prepare candidate prompts for rollout
    candidate_prompts = []
    for beam in active_beams:
        chat = prepare_chat_template(example, system_prompt)
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        ).rstrip(tokenizer.eos_token).rstrip()
        prompt += beam['sequence']
        candidate_prompts.append(prompt)
    
    token_stats = {
        "input": sum(len(tokenizer(p)["input_ids"]) for p in candidate_prompts),
        "output": 0
    }
    
    # Step 2: Generate candidate next-step outputs
    sampling_params_step = SamplingParams(
        max_tokens=2048,
        n=num_rollout,
        logprobs=0,
        temperature=0.7,
        stop=["\n", "<end_of_reasoning>"]
    )
    
    step_outputs = model.generate(candidate_prompts, sampling_params_step)
    
    # Step 3: Collect all rollout responses
    all_rollout_responses = []
    for beam_idx, beam_outputs in enumerate(step_outputs):
        beam = active_beams[beam_idx]
        beam_sequence = beam['sequence']
        prob_before = beam['half_prob']
        
        for output in beam_outputs.outputs:
            response_text = output.text.strip()
            # Check if response contains end token or is empty
            is_finished = (response_text == "" or 
                        "<end_of_reasoning>" in response_text)
            
            if response_text == "":
                # Empty response - treat as finished with current quality
                if beam_sequence == "The reasoning steps are:\n\n":
                    continue
                all_rollout_responses.append({
                    'sequence': beam_sequence,
                    'logprob': prob_before,
                    'finished': True,
                    'beam_idx': beam_idx
                })
                print('Warning: response is empty')
                continue
            
            # Calculate logprob for this step
            step_logprob = output.cumulative_logprob / (len(output.token_ids) + 1e-8)
            new_sequence = beam_sequence + response_text
            if not is_finished:
                new_sequence += "\n"
            
            all_rollout_responses.append({
                'sequence': new_sequence,
                'logprob': step_logprob,
                'finished': is_finished,
                'beam_idx': beam_idx
            })
            token_stats["output"] += len(output.token_ids)
    
    # Step 4: Separate finished and unfinished responses
    finished_responses = [r for r in all_rollout_responses if r['finished']]
    unfinished_responses = [r for r in all_rollout_responses if not r['finished']]
    
    print(f"Finished: {len(finished_responses)}, Unfinished: {len(unfinished_responses)}")
    
    # Step 5: Handle different scenarios
    candidate_paths = []
    
    # Scenario 1: All paths are finished
    if not unfinished_responses:
        print("All paths finished, returning top candidates")
        finished_sorted = sorted(finished_responses, key=lambda x: x['logprob'], reverse=True)
        for i, response in enumerate(finished_sorted[:step_beam_size]):
            candidate_paths.append({
                'sequence': response['sequence'],
                'final_quality': response['logprob'],
                'advantage': active_beams[response['beam_idx']].get('advantage', 0.0),
                'half_prob': response['logprob'],
                'finished': True
            })
        return candidate_paths, token_stats, True
    
    # Scenario 2: Mix of finished and unfinished paths - prioritize finished paths
    # Sort finished paths by quality
    finished_sorted = sorted(finished_responses, key=lambda x: x['logprob'], reverse=True)
    
    # Keep at most 60% of beam size as finished paths
    max_finished_to_keep = min(len(finished_sorted), max(1, int(step_beam_size * 0.6)))
    
    # Add top finished paths to candidates
    for i, response in enumerate(finished_sorted[:max_finished_to_keep]):
        candidate_paths.append({
            'sequence': response['sequence'],
            'final_quality': response['logprob'],
            'advantage': active_beams[response['beam_idx']].get('advantage', 0.0),
            'half_prob': response['logprob'],
            'finished': True
        })
    
    # Step 6: Process unfinished paths with foresight
    remaining_slots = step_beam_size - len(candidate_paths)
    
    if remaining_slots > 0 and unfinished_responses:
        # Apply sigma-based pruning to unfinished responses
        unfinished_logprobs = [r['logprob'] for r in unfinished_responses]
        if len(unfinished_logprobs) > 1:
            mean_logprob = np.mean(unfinished_logprobs)
            std_logprob = np.std(unfinished_logprobs)
            threshold = mean_logprob + sigma_rate * std_logprob
            
            # Filter based on threshold
            filtered_unfinished = [r for r in unfinished_responses if r['logprob'] >= threshold]
            if len(filtered_unfinished) == 0:
                filtered_unfinished = unfinished_responses[:1]
        else:
            filtered_unfinished = unfinished_responses
        
        print(f"Filtered unfinished paths: {len(filtered_unfinished)}")
        
        # Prepare foresight prompts for filtered unfinished paths
        foresight_prompts = []
        for response in filtered_unfinished:
            chat = prepare_chat_template(example, system_prompt)
            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            ).rstrip(tokenizer.eos_token).rstrip()
            prompt += response['sequence']
            foresight_prompts.append(prompt)
        
        token_stats["input"] += sum(len(tokenizer(p)["input_ids"]) for p in foresight_prompts)
        
        # Generate foresight completions and calculate advantages
        if foresight_prompts:
            sampling_params_foresight = SamplingParams(
                max_tokens=2048,
                n=1,
                logprobs=0,
                stop=["<end_of_reasoning>"]
            )
            foresight_outputs = model.generate(foresight_prompts, sampling_params_foresight)
            token_stats["output"] += sum(len(output.outputs[0].token_ids) for output in foresight_outputs)
            
            # Calculate advantages by comparing foresight quality to baseline
            path_advantages = []
            for i, (response, foresight_output) in enumerate(zip(filtered_unfinished, foresight_outputs)):
                foresight_logprob = foresight_output.outputs[0].cumulative_logprob / (
                    len(foresight_output.outputs[0].token_ids) + 1e-8
                )
                
                beam_baseline = active_beams[response['beam_idx']]['half_prob']
                advantage = foresight_logprob - beam_baseline
                
                path_advantages.append({
                    'response': response,
                    'foresight_quality': foresight_logprob,
                    'advantage': advantage
                })
            
            # Select best unfinished paths based on advantages (deterministic top-k)
            path_advantages.sort(key=lambda x: x['advantage'], reverse=True)
            
            if remaining_slots >= len(path_advantages):
                selected_unfinished = path_advantages
            else:
                selected_unfinished = path_advantages[:remaining_slots]
            
            # Add selected unfinished paths to candidate list
            for path_info in selected_unfinished:
                response = path_info['response']
                candidate_paths.append({
                    'sequence': response['sequence'],
                    'final_quality': path_info['foresight_quality'],
                    'advantage': path_info['advantage'],
                    'half_prob': response['logprob'],
                    'finished': False
                })
    print(f"Final candidate paths: {len(candidate_paths)}")
    
    # Check if all paths are finished
    all_finished = all(path['finished'] for path in candidate_paths)
    
    return candidate_paths, token_stats, all_finished
    

def generate_final_response(example, system_prompt, active_beams):
    """Generate final response from the best beam."""
    token_stats = {"input": 0, "output": 0}
    
    # Separate finished and unfinished beams
    finished_beams = [beam for beam in active_beams if beam.get('finished', False)]
    unfinished_beams = [beam for beam in active_beams if not beam.get('finished', False)]
    
    all_combined_responses = []
    all_final_qualities = []
    
    # Process finished beams - use sequences directly
    for beam in finished_beams:
        all_combined_responses.append(beam['sequence'])
        all_final_qualities.append(beam['final_quality'])
    
    # Process unfinished beams - generate completions
    if unfinished_beams:
        all_inputs = []
        for beam in unfinished_beams:
            chat = prepare_chat_template(example, system_prompt)
            final_prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            ).rstrip(tokenizer.eos_token).rstrip()
            final_prompt += beam['sequence']
            all_inputs.append(final_prompt)
            token_stats["input"] += len(tokenizer(final_prompt)["input_ids"])


        sampling_params = SamplingParams(
            max_tokens=50000, 
            n=1, 
            logprobs=0, 
            stop=["<end_of_reasoning>"]
        )
        outputs = model.generate(all_inputs, sampling_params)
        
        for beam_idx, (beam, beam_outputs) in enumerate(zip(unfinished_beams, outputs)):
            output = beam_outputs.outputs[0]
            response = output.text.strip()
            
            if response != "":
                logprob = output.cumulative_logprob / (len(output.token_ids) + 1e-8)
                final_quality = logprob
            else:
                final_quality = beam['final_quality']
            
            combined_response = beam['sequence'] + response
            all_combined_responses.append(combined_response)
            all_final_qualities.append(final_quality)
            token_stats["output"] += len(output.token_ids)
    
    # Select the best response based on quality
    if all_final_qualities:
        best_idx = np.argmax(all_final_qualities)
        final_response = all_combined_responses[best_idx]
    else:
        # Fallback: return first beam's sequence if no valid responses
        final_response = active_beams[0]['sequence']
    
    return final_response, token_stats, all_combined_responses

def process_example(example, system_prompt):
    """Process a single example through the beam search pipeline."""
    token_stats = {"input": 0, "output": 0}

    # Initialize beams with starting sequence
    initial_sequence = "The reasoning steps are:\n\n"
    active_beams = [{'sequence': initial_sequence, 'half_prob': -1000, 'advantage': -1000} 
                    for _ in range(step_beam_size)]
    
    f_best_prev = -np.inf  # Quality of the best beam from the previous step

    for step in range(num_foresight):
        active_beams_before = copy.deepcopy(active_beams)
        candidate_paths, step_token_stats, finished_flag = process_step(example, system_prompt, active_beams)
        token_stats["input"] += step_token_stats["input"]
        token_stats["output"] += step_token_stats["output"]
        if not candidate_paths:
            print(f"Stopping at step {step} due to no valid candidate paths.")
            break

        # Check if all paths are finished
        if finished_flag:
            print(f"Stopping at step {step}: All paths are finished.")
            active_beams = candidate_paths
            break
        
        active_beams = candidate_paths
        
        # Sort unfinished beams by advantage for convergence check
        active_beams_compare = sorted(
            [beam for beam in active_beams if beam["finished"] is False],
            key=lambda x: x["advantage"],
            reverse=True
        )
        
        # Convergence check: early stopping based on advantage
        f_best_now = active_beams_compare[0]['advantage']
        if step > minimum_foresight_steps:
            one_step_advantage = f_best_now
            print(f"One step advantage: {one_step_advantage:.4f}")
            if one_step_advantage <= EARLY_STOPPING_ADVANTAGE:
                print(f"Early stopping at step {step}: Advantage {one_step_advantage:.4f} <= {EARLY_STOPPING_ADVANTAGE}")
                active_beams = active_beams_before
                break
    final_response, final_token_stats, all_responses = generate_final_response(example, system_prompt, active_beams)
    
    print(f"Generated text: {final_response}")
    token_stats["input"] += final_token_stats["input"]
    token_stats["output"] += final_token_stats["output"]
    print(f"Input tokens: {token_stats['input']}")
    print(f"Output tokens: {token_stats['output']}")

    return {"response": final_response, "token_stats": token_stats, "all_responses": all_responses}


MATH_COT_4_SHOT = """
Example 1:
The question is : Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?
The reasoning steps are:

The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ in the complex plane is given by the formula $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$.
In this case, Joe's point is $(1,2)$ and Gracie's point is $(-1,1)$.
So the distance between their points is $\\sqrt{((-1)-(1))^2+((1)-(2))^2}=\\sqrt{(-2)^2+(-1)^2}=\\sqrt{4+1}=\\sqrt{5}$.
Therefore, Gracie and Joe's points are $\\boxed{\\sqrt{5}}$ units apart.
The answer is: \\boxed{\\sqrt{5}}<end_of_reasoning>


Example 2:
The question is : Convert $10101_3$ to a base 10 integer.
The reasoning steps are:

$10101_3 = 1 \\cdot 3^4 + 0 \\cdot 3^3 + 1 \\cdot 3^2 + 0 \\cdot 3^1 + 1 \\cdot 3^0 = 81 + 9 + 1 = \\boxed{91}$.
The answer is: \\boxed{91}<end_of_reasoning>


Example 3:
The question is :
The points $(x, y)$ represented in this table lie on a straight line. The point $(28, t)$ lies on the same line. What is the value of $t?$ \\begin{tabular}{c|c}\n$x$ & $y$ \\\\ \\hline\n1 & 7 \\\\\n3 & 13 \\\\\n5 & 19 \\\\\n\\end{tabular}

The reasoning steps are:
The slope of a line passing through two points $(x_1, y_1)$ and $(x_2, y_2)$ is given by $\\frac{y_2 - y_1}{x_2 - x_1}$.
Using the points $(1, 7)$ and $(5, 19)$ from the table, we find that the slope of the line passing through these points is $\\frac{19 - 7}{5 - 1} = \\frac{12}{4} = 3$.
Since the point $(28, t)$ lies on the same line, the slope of the line passing through $(28, t)$ and $(5, 19)$ is also $3$.
Using the slope-intercept form of a line, $y = mx + b$, where $m$ is the slope and $b$ is the $y$-intercept, we can find the equation of the line passing through $(5, 19)$ with a slope of $3$.
Substituting the coordinates of the point $(5, 19)$ into the equation, we have $19 = 3(5) + b$, which gives us $b = 19 - 15 = 4$.
Therefore, the equation of the line passing through these two points is $y = 3x + 4$.
Substituting $x = 28$ into this equation, we can find the value of $t$:
$t = 3(28) + 4 = 84 + 4 = \\boxed{88}$.
The answer is: \\boxed{88}<end_of_reasoning>


Example 4:
The question is :
Five socks, colored blue, brown, black, red, and purple are in a drawer. In how many different ways can we choose three socks from the drawer if the order of the socks does not matter?

The reasoning steps are:
This is a combination problem, since the order of the socks does not matter.
We want to choose 3 out of the 5 socks, so we can use the formula for combinations:
$\\binom{n}{k}=\\dfrac{n!}{k!(n-k)!}$.
In this case, $n=5$ (the total number of socks) and $k=3$ (the number of socks to choose).
Plugging in the values, we get $\\binom{5}{3}=\\dfrac{5!}{3!(5-3)!}=\\dfrac{5!}{3!2!}=\\dfrac{5\\times4\\times3\\times2\\times1}{3\\times2\\times1\\times2\\times1}=\\dfrac{5\\times4}{2\\times1}=\\boxed{10}$.
The answer is: \\boxed{10}<end_of_reasoning>

""".strip()
LOGIC_MRC_COT_4_SHOT = """
Example 1:

Passage: A professional baseball team manager, in order to have the funds to sign a new second-baseman, discreetly arranged to trade one of the most popular outfielders on the team for a lesser-known player and an undisclosed amount of money. The manager secretly considered the outfielder to be overrated and overpaid. Reporters forcefully criticized the trade, arguing that the team had lost its best player and that the manager had violated his fiduciary duties to the team and the fans. A few weeks after being traded, the outfielder was retraded, for twice the value received by the original team manager. Plainly, the outfielder' s price shows that the reporters' criticism of the manager' s action was accurate.
Question: The reasoning in the argument is vulnerable to the criticism that the argument does which one of the following?
A. The argument bases its conclusion on what the best decision is for the present on uncertain projections about what the best decision will be for the future.
B. The argument rejects a well-established way of achieving an end without explaining why an unconventional way is better suited for achieving the end.
C. The argument ignores the opinions of expert reporters in the field of baseball when there is no superior source of information.
D. The argument bases its conclusion on facts that could, considering the circumstances, have been the product of circumstances other than those presumed by the argument's proponents.

The reasoning steps are:

The passage argues that the reporters' criticism of the manager's decision was accurate because the outfielder was retraded for twice the value. 
However, this change in the outfielder's value could be due to various factors other than the manager's poor decision-making, such as market fluctuations, the outfielder's performance, or the demands of other teams. 
The argument does not consider these other possible circumstances when concluding that the manager's decision was wrong.
The answer is D.<end_of_reasoning>


Example 2:

Passage: Quality control investigator: Upon testing samples of products from our supplier that were sent by our field inspectors from various manufacturing locations, our laboratory discovered that over 20 percent of the samples were defective. Since our supplier is contractually required to limit the rate of defects among items it manufactures for us to below 5 percent, it has violated its contract with us.
Question: The reasoning in the quality control investigator's argument is flawed in that the argument
A. presumes, without providing justification, that the field inspectors were just as likely to choose a defective item for testing as they were to choose a nondefective item
B. presumes, without providing justification, that the field inspectors made an equal number of visits to each of the various manufacturing sites of the supplier
C. overlooks the possibility that the field inspectors tend to choose items for testing that they suspect are defective
D. bases its conclusion on too small a sample of items tested by the laboratory

The reasoning steps are:

The reasoning in the quality control investigator's argument is flawed because it overlooks the possibility that the field inspectors tend to choose items for testing that they suspect are defective.
This means that the 20 percent defect rate discovered by the laboratory might not accurately represent the overall defect rate among the items manufactured by the supplier.
The answer is: C.<end_of_reasoning>


Example 3:

Passage: The Levant -- the area that borders the eastern Mediterranean-was heavily populated in prehistoric times. The southern Levant was abandoned about 6, 000 years ago, although the northern Levant, which shared the same climate, remained heavily populated. Recently archaeologists have hypothesized that the sudden depopulation in the southern Levant was due to an economic collapse resulting from deforestation.
Question: If the statements above are true and the archaeologists' hypothesis is correct, which one of the following CANNOT be true?
A. The sheep and goats herded by the peoples of the southern Levant until 6, 000 years ago grazed extensively on the seedlings and saplings of indigenous tree species.
B. Carbon dating of organic remains from the southern Levant reliably demonstrates that there were no forests present in that area prior to 6, 000 years ago.
C. Organic remains from the northern Levant reliably indicate that tree species flourished there without interruption during the period when the southern Levant was being abandoned.
D. Since there are few traces of either quarried stone or of mud brick in buildings excavated in the southern Levant, it is likely that the buildings built there prior to 6, 000 years ago were made almost entirely of timber.

The reasoning steps are:

If there were no forests present in the southern Levant prior to 6, 000 years ago, it would not make sense for the cause of the economic collapse to be deforestation, as the area would have already been deforested. 
This contradicts the archaeologists' hypothesis, making option B the correct answer."
The answer is: B.<end_of_reasoning>


Example 4:

Passage: The most successful economies have been, and will continue to be, those that train as many people as possible in the human skills required to research, to develop, and to apply new technology. Japan is a model for this sort of training effort. Europe as a whole is in a weaker position: there is a shortage of skilled labor trained to use the new technologies, and there are not enough scientists able to develop and apply the technology. However, even in Japan there is a shortage of technically qualified people, and, like most European countries, Japan has far too many workers qualified to perform only menial tasks.
Question: Which one of the following can be properly inferred from the passage?
A. To be economically more successful, Europe needs to train more people in the new technologies.
B. Japan's successful economy depends upon an uncommonly narrow base of highly skilled labor.
C. Japan is not the best country against which to measure a country's economic success.
D. European countries have economies that are more successful than those of most other countries.

The reasoning steps are:

To be economically more successful, Europe needs to train more people in the new technologies.
The passage states that the most successful economies are those that train as many people as possible in the required human skills.
It also mentions that Europe is in a weaker position due to a shortage of skilled labor and scientists in new technologies.
Therefore, we can infer that training more people in new technologies would help Europe to become economically more successful.
The answer is A.<end_of_reasoning>
""".strip()

GSM_COT_8_SHOT = """
Example 1:
The question is : There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
The reasoning steps are:

There are 15 trees originally. Then there were 21 trees after some more were planted. 
So there must have been 21 - 15 = 6.
The answer is: \\boxed{6}.<end_of_reasoning>


Example 2:
The question is: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
The reasoning steps are:

There are originally 3 cars. 
2 more cars arrive. 
3 + 2 = 5.
The answer is: \\boxed{5}.<end_of_reasoning>


Example 3:
The question is: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
The reasoning steps are:

Originally, Leah had 32 chocolates. Her sister had 42. 
So in total they had 32 + 42 = 74. 
After eating 35, they had 74 - 35 = 39.
The answer is: \\boxed{39}.<end_of_reasoning>


Example 4:
The question is: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
The reasoning steps are:

Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8.
The answer is: \\boxed{8}.<end_of_reasoning>


Example 5:
The question is: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
The reasoning steps are:

Shawn started with 5 toys. 
If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. 
The answer is: \\boxed{9}.<end_of_reasoning>


Example 6:
The question is: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
The reasoning steps are:

There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added. 9 + 20 is 29.
The answer is: \\boxed{29}.<end_of_reasoning>


Example 7:
The question is: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
The reasoning steps are:

Michael started with 58 golf balls.
After losing 23 on tuesday, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = 33 golf balls.
The answer is: \\boxed{33}.<end_of_reasoning>


Example 8:
The question is: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
The reasoning steps are:

Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left. 23 - 15 is 8.
The answer is: \\boxed{8}.<end_of_reasoning>

""".strip()

# Main execution loop
total_input_tokens = 0
total_output_tokens = 0

# Clear output files
open(output_path, "w").close()

for i, example in enumerate(test_data):
    print(f"Processing example {i+1}/{len(test_data)}...")
    system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{GSM_COT_8_SHOT}"
    result = process_example(example, system_prompt)
    total_input_tokens += result["token_stats"]["input"]
    total_output_tokens += result["token_stats"]["output"]

    # Save results
    output_result = {
        "id": i,
        "response": result['response'],
        'all_responses': result['all_responses']
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(output_result) + "\n")
    print(f"Finished example {i+1}. Current total tokens (in/out): {total_input_tokens}/{total_output_tokens}")
    
    # Save token statistics
    with open(output_path2, "a") as f:
        f.write(json.dumps(result['token_stats']) + "\n")