from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import pandas as pd

pseudo_labeled_reviews = pd.read_parquet('../datafiles/pseudo_labeled_corpus.parquet', engine='pyarrow')
pseudo_labeled_reviews = pseudo_labeled_reviews[pseudo_labeled_reviews['relevancy_label']=='maybe-privacy']
pseudo_labeled_reviews = pseudo_labeled_reviews.reset_index(drop=True)

best_llm = ''
best_F1_score = 0

# huggingface login
access_token = "<huggingface_token>"
login(access_token)

models = ['meta-llama/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct', 'tiiuae/falcon-7b-instruct', 'mistralai/Mistral-7B-Instruct-v0.3']

domain_specific_hypotheses = [
    "User data being linked across different services.",
    "Online user activities from various platforms can be connected.",
    "Anonymized user data could be used to reveal their identity.",
    "Unique digital user data could lead to personal identification.",
    "User is unable to deny their online actions.",
    "User is concerned about the permanent storage of their digital transactions.",
    "User is concerned about others detecting their use of sensitive online services.",
    "User presence on certain platforms could be discovered from anonymized data.",
    "User device's communication patterns reveal private information.",
    "User personal data intercepted during transmission.",
    "Unauthorized access to user's private information.",
    "The user is not aware of how and why their data is being collected, processed, stored, and shared.",
    "The user is concerned about the processing or storing of their personal data against regulations or privacy policies.",
    "The user is facing a privacy issue.",
    "Personal user information is collected from other sources.",
    "The user is concerned about protecting their personal data.",
    "A data anonymity topic is discussed.",
    "The app exposes a private aspect of the user life.",
    "User data is being exploited for other purposes.",
    "Data sharing with third parties is discussed.",
    "A data privacy topic is discussed.",
]



def analyze_review(review, model, tokenizer):
    prompt = [
        {
            'role': 'system',
            'content': f'''You are a scholarly researcher and your task is to annotate the data. 
You will receive an app review and you have to annotate it with a yes or no label based on the mental health domain-specific privacy hypothesis provided below.
If the review satisfies any of the hypothesis then annotate it with a yes label otherwise annotate it with a no label.
Privacy Hypotheses:
    - {'\n - '.join(domain_specific_hypotheses)}
            '''
        },
        {
            'role': 'user',
            'content': f'''App Review: {review}
Does this app review satisfies any of the hypothesis? Respond with yes or no'''
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def calculate_metrics(data):
    P, R, F1 = 0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for key in data[data['Privacy-related']==1]['llm_response'].value_counts().keys():
        if key=='yes':
            TP += data[data['Privacy-related']==1]['llm_response'].value_counts().get(key)
        else:
            FN += data[data['Privacy-related']==1]['llm_response'].value_counts().get(key)
    for key in data[data['Privacy-related']==0]['llm_response'].value_counts().keys():
        if key=='yes':
            FP += data[data['Privacy-related']==1]['llm_response'].value_counts().get(key)
        else:
            TN += data[data['Privacy-related']==1]['llm_response'].value_counts().get(key)

    if TP>0:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = (2*P*R) / (P+R)

    return {'P': P, 'R': R, 'F1': F1}


def llm_inference(reviews, model, tokenizer):
    llm_response = []
    for review in reviews:
        res = analyze_review(review, model, tokenizer)
        llm_response.append(res)
    labeled_df = pd.concat([pseudo_labeled_reviews, pd.DataFrame({'llm_response': llm_response})], axis=1)
    metrics = calculate_metrics(labeled_df)
    return metrics


def evaluate_llms():
    reviews = pseudo_labeled_reviews['clean_content'].to_list()
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        metrics = llm_inference(reviews, model, tokenizer)
        print(f'Model: {model_name}, Metrics: {metrics}') # record the results
        if metrics['F1']>best_F1_score:
            best_F1_score=metrics['F1']
            best_llm=model_name


def baseline():
    P = len(pseudo_labeled_reviews[pseudo_labeled_reviews['Privacy-related']==1])/len(pseudo_labeled_reviews)
    R = 0.5
    F1 = (2*P*R)/(P+R)
    best_F1_score=F1
    best_llm='random-classifier'


baseline()
evaluate_llms()

print(best_llm) # best LLM