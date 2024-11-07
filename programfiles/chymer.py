from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

reviews = pd.read_csv('../datafiles/MH_12star_reviews.csv')
reviews = reviews.query("app in ['calm', 'headspace', 'sanvello', 'talkspace', 'shine']").reset_index(drop=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

nli_model = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
llm = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

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


def apply_heuristics_domainSpecificHypotheses(entailment_scores):
    labels = []
    for score_l in entailment_scores:
        if score_l is not None and len(score_l)>0:
            sorted_scores = sorted(score_l, reverse=True)
            if sorted_scores[0]>=0.85 or sorted_scores[2]>=0.75 or sorted_scores[4]>=0.7:
                label = 'maybe-privacy'
            else:
                label = 'maybe-not-privacy'
        else:
            label = 'maybe-not-privacy'
        labels.append(label)
    return labels


def execute_nli(hypotheses, reviews, model, tokenizer):
    entailment_scores = []
    for idx in range(0, len(reviews)):
        review = reviews[idx]
        scores = []
        if review and isinstance(review, str):
            try:
                for hpt in hypotheses:
                    input = tokenizer(review, hpt, truncation=True, return_tensors="pt")
                    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
                    prediction = torch.softmax(output["logits"][0], -1).tolist()
                    scores.append(prediction[0])
            except Exception:
                print(f'some error occured for index: {idx}')
        entailment_scores.append(scores)
    return entailment_scores


def nli_inference_domainSpecific():
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model)
    model.to(device)
    reviews = reviews['clean_content'].to_list()
    entailment_scores = execute_nli(domain_specific_hypotheses, reviews, model, tokenizer)
    nli_annotations = apply_heuristics_domainSpecificHypotheses(entailment_scores)
    pseudo_labeled_df = pd.concat([reviews, pd.DataFrame({'relevancy_label': nli_annotations})], axis=1)
    return pseudo_labeled_df


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


def llm_inference(pseudo_labeled_reviews):
    tokenizer = AutoTokenizer.from_pretrained(llm)
    model = AutoModelForCausalLM.from_pretrained(
        llm,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    reviews = pseudo_labeled_reviews['clean_content'].to_list()
    llm_response = []
    for review in reviews:
        res = analyze_review(review, model, tokenizer)
        llm_response.append(res)
    labeled_df = pd.concat([pseudo_labeled_reviews, pd.DataFrame({'llm_response': llm_response})], axis=1)
    return labeled_df

pseudo_labeled_corpus = nli_inference_domainSpecific()
potential_privacy_reviews = pseudo_labeled_corpus[pseudo_labeled_corpus['relevancy_label']=='maybe-privacy']
potential_privacy_reviews = potential_privacy_reviews.reset_index(drop=True)

llm_classified_dataset = llm_inference(potential_privacy_reviews)
privacy_reviews = llm_classified_dataset[llm_classified_dataset['llm_response']=='yes']
privacy_reviews.to_csv('../datafiles/chymer_classified_privacy_reviews.csv')