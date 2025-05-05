import os
import pandas as pd
import joblib
from tqdm import tqdm
from transformers import pipeline

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
MODEL_PATH = "email_model.pkl"
DATA_PATH = "emails.csv"
CANDIDATE_LABELS = ["Work", "Personal", "Social", "Promotions", "Updates", "Spam", "Finance", "Health", "Education"]


def authenticate_gmail():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def fetch_paged(service, max_items, q="label:inbox"):
    """Fetch up to max_items messages matching q, paging through results."""
    collected, page_token = [], None
    pbar = tqdm(total=max_items, desc="Fetching emails")
    while len(collected) < max_items:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q=q, maxResults=min(500, max_items - len(collected)), pageToken=page_token)
            .execute()
        )
        msgs = resp.get("messages", [])
        if not msgs:
            break
        for m in msgs:
            md = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
            hdrs = md["payload"].get("headers", [])
            subj = next((h["value"] for h in hdrs if h["name"] == "Subject"), "")
            frm = next((h["value"] for h in hdrs if h["name"] == "From"), "")
            snip = md.get("snippet", "")
            collected.append({"id": m["id"], "subject": subj, "from": frm, "body": snip})
            pbar.update(1)
            if len(collected) >= max_items:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    pbar.close()
    return collected


def get_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def ensure_label(service, label_name):
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for lab in labels:
        if lab["name"] == label_name:
            return lab["id"]
    body = {"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
    return service.users().labels().create(userId="me", body=body).execute()["id"]


def load_existing_ids():
    if not os.path.exists(DATA_PATH):
        return set()
    return set(pd.read_csv(DATA_PATH)["id"].astype(str))


def option_train(service):
    existing = load_existing_ids()
    N = int(input("How many emails to fetch & zero-shot label for training? "))
    all_emails = fetch_paged(service, N, q="label:inbox")
    new = [e for e in all_emails if e["id"] not in existing]
    if not new:
        print("❌ No new emails to label.")
        return

    clf0 = get_zero_shot_classifier()
    for e in tqdm(new, desc="Zero‑shot labeling"):
        txt = f"{e['subject']} {e['body']} {e['from']}"
        res = clf0(txt, CANDIDATE_LABELS)
        e["label"] = res["labels"][0]

    df_new = pd.DataFrame(new)
    if existing:
        df_old = pd.read_csv(DATA_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Appended {len(new)} new labeled emails to {DATA_PATH}")

    # Train sklearn model
    X = df["subject"].fillna("") + " " + df["body"].fillna("") + " " + df["from"].fillna("")
    y = df["label"]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    print("✅ Sklearn model trained & saved.")


def option_label(service):
    if not os.path.exists(MODEL_PATH):
        print("❌ No trained model found. Run option 1 first.")
        return

    existing = load_existing_ids()
    M = int(input("How many recent emails to label? "))
    all_emails = fetch_paged(service, M, q="label:inbox")
    new = [e for e in all_emails if e["id"] not in existing]
    if not new:
        print("❌ No new emails to label.")
        return

    pipe = joblib.load(MODEL_PATH)
    label_ids = {}

    for e in tqdm(new, desc="Predicting & labeling"):
        txt = f"{e['subject']} {e['body']} {e['from']}"
        lbl = pipe.predict([txt])[0]
        e["label"] = lbl
        lab_id = label_ids.setdefault(lbl, ensure_label(service, lbl))
        service.users().messages().modify(
            userId="me", id=e["id"], body={"addLabelIds": [lab_id], "removeLabelIds": ["UNREAD"]}
        ).execute()

    df_new = pd.DataFrame(new)
    df_old = pd.read_csv(DATA_PATH)
    df = pd.concat([df_old, df_new], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Labeled & applied Gmail labels to {len(new)} emails")


if __name__ == "__main__":
    svc = authenticate_gmail()
    print("1) Train model (zero‑shot → sklearn)")
    print("2) Label new emails in Gmail")
    choice = input("Choose 1 or 2: ").strip()
    if choice == "1":
        option_train(svc)
    elif choice == "2":
        option_label(svc)
    else:
        print("❌ Invalid choice.")
