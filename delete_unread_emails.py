from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from tqdm import tqdm
import os.path
import time

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


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
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def delete_unread_emails(service, query):
    total_deleted = 0
    next_page_token = None

    while True:
        results = service.users().messages().list(userId="me", q=query, pageToken=next_page_token).execute()

        messages = results.get("messages", [])
        if not messages:
            break

        print(f"Found {len(messages)} messages. Deleting...")

        for msg in tqdm(messages, desc="Deleting", unit="msg"):
            msg_id = msg["id"]
            service.users().messages().trash(userId="me", id=msg_id).execute()
            total_deleted += 1
            time.sleep(0.01)  # Optional: avoid hitting API rate limits

        next_page_token = results.get("nextPageToken")
        if not next_page_token:
            break

    print(f"\nTotal messages moved to trash: {total_deleted}")


if __name__ == "__main__":
    gmail_service = authenticate_gmail()
    delete_unread_emails(gmail_service, query="is:unread category:updates")
