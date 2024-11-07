from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

labeled_reviews = pd.read_csv('../datafiles/privacy_gt_dataset.csv')

labeled_reviews_mh_domain = labeled_reviews.query("app in ['calm', 'headspace', 'sanvello', 'talkspace', 'shine']").reset_index(drop=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

nli_models = ['MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', 'google-t5/t5-base', 'cross-encoder/nli-roberta-base', 'FacebookAI/roberta-large-mnli']

generic_hypotheses = [
    "The user is facing a data surveillance issue.",
    "The user is forced to provide information.",
    "Personal user information is collected from other sources.",
    "The user is concerned about protecting their personal data.",
    "A data anonymity topic is discussed.",
    "The user is concerned about the purposes of personal data access.",
    "The user wants to correct their personal information.",
    "A breach of data confidentiality is discussed.",
    "Personal data disclosure is discussed.",
    "The app exposes a private aspect of the user life.",
    "User's data has been made accessible to public.",
    "A data blackmailing issue is discussed.",
    "User data is being exploited for other purposes.",
    "False data is presented about the user.",
    "Unwanted intrusion to personal info is discussed.",
    "Intrusion by the government to the user's life is discussed.",
    "Opting out from personal data collection is discussed.",
    "More access than needed is required.",
    "The reason for data access is not provided.",
    "Too much personal data is collected.",
    "The data is being used for unexpected purposes.",
    "Data sharing with third parties is discussed.",
    "User choice for personal data collection is discussed.",
    "User did not allow access to their personal data.",
    "A data privacy topic is discussed.",
    "Protecting user's personal data is discussed.",
    "This is about a privacy feature.",
    "The user is facing a privacy issue.",
    "The user likes that data privacy is provided.",
    "The user wants privacy.",
    "The app has privacy features."
]

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

best_F1_score = 0
best_nli_model = ''

def apply_heuristics_genericHypotheses(entailment_scores):
    labels = []
    for score_l in entailment_scores:
        if score_l is not None and len(score_l)>0:
            sorted_scores = sorted(score_l, reverse=True)
            if sorted_scores[0]<0.4:
                label='maybe-not-privacy'
            elif sorted_scores[0]>0.8 or sorted_scores[2]>0.7 or sorted_scores[4]>0.6 or sorted_scores[6]>0.5:
                label = 'maybe-privacy'
            else:
                label = 'undetermined'
        else:
            label = 'undetermined'
        labels.append(label)
    return labels

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


def calculate_metrics(data):
    P, R, F1 = 0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for key in data[data['Privacy-related']==1]['relevancy_label'].value_counts().keys():
        if key=='maybe-privacy':
            TP += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)
        else:
            FN += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)
    for key in data[data['Privacy-related']==0]['relevancy_label'].value_counts().keys():
        if key=='maybe-privacy':
            FP += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)
        else:
            TN += data[data['Privacy-related']==1]['relevancy_label'].value_counts().get(key)

    if TP>0:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = (2*P*R) / (P+R)

    return {'P': P, 'R': R, 'F1': F1}


def nli_inference_genericHypotheses():
    for model_name in nli_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        reviews = labeled_reviews_mh_domain['clean_content'].to_list()
        entailment_scores = execute_nli(generic_hypotheses, reviews, model, tokenizer)
        nli_annotations = apply_heuristics_genericHypotheses(entailment_scores)
        pseudo_labeled_df = pd.concat([labeled_reviews_mh_domain, pd.DataFrame({'relevancy_label': nli_annotations})], axis=1)
        metrics = calculate_metrics(pseudo_labeled_df)
        print(f'Model: {model_name}, Metrics: {metrics}') # record the results
        if metrics['F1']>best_F1_score:
            best_F1_score=metrics['F1']
            best_nli_model=model_name

nli_inference_genericHypotheses()


def nli_inference_domainSpecific():
    tokenizer = AutoTokenizer.from_pretrained(best_nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(best_nli_model)
    model.to(device)
    reviews = labeled_reviews_mh_domain['clean_content'].to_list()
    entailment_scores = execute_nli(domain_specific_hypotheses, reviews, model, tokenizer)
    nli_annotations = apply_heuristics_domainSpecificHypotheses(entailment_scores)
    pseudo_labeled_df = pd.concat([labeled_reviews_mh_domain, pd.DataFrame({'relevancy_label': nli_annotations})], axis=1)
    metrics = calculate_metrics(pseudo_labeled_df)
    print(f'Model: {best_nli_model}, Metrics: {metrics}') # record the results
    return metrics


def compare_genericHypo_domainSpecificHypo():
    domainSpecific_metrics = nli_inference_domainSpecific()
    if domainSpecific_metrics['F1']>best_F1_score:
        best_F1_score=domainSpecific_metrics['F1']
        best_hypotheses='domain-specific'
    else:
        best_hypotheses='generic'
    return best_hypotheses

best_hypotheses = compare_genericHypo_domainSpecificHypo()
print(f'Best set of hypotheses: {best_hypotheses}') # record the results


# create pseudo-labeled corpus with the best nli model, best set of hypotheses and corresponding heuristics
tokenizer = AutoTokenizer.from_pretrained(best_nli_model)
model = AutoModelForSequenceClassification.from_pretrained(best_nli_model)
model.to(device)
reviews = labeled_reviews_mh_domain['clean_content'].to_list()
if best_hypotheses=='generic':
    entailment_scores = execute_nli(generic_hypotheses, reviews, model, tokenizer)
    nli_annotations = apply_heuristics_genericHypotheses(entailment_scores)
else:
    entailment_scores = execute_nli(domain_specific_hypotheses, reviews, model, tokenizer)
    nli_annotations = apply_heuristics_domainSpecificHypotheses(entailment_scores)
    
pseudo_labeled_df = pd.concat([labeled_reviews_mh_domain, pd.DataFrame({'relevancy_label': nli_annotations})], axis=1)
pseudo_labeled_df.to_parquet('../datafiles/pseudo_labeled_corpus.csv', engine='pyarrow')