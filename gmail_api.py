from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import base64
import os
import pickle
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def authenticate_gmail():
    creds = None
    if os.path.exists("token.pkl"):
        with open("token.pkl", "rb") as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pkl", "wb") as token:
            pickle.dump(creds, token)
    
    return build("gmail", "v1", credentials=creds)

def get_latest_email():
    service = authenticate_gmail()

    # Fetch latest emails (up to 5 for sorting)
    results = service.users().messages().list(userId="me", maxResults=5, labelIds=["INBOX"], q="").execute()
    messages = results.get("messages", [])

    if not messages:
        print("No new emails found.")
        return None

    latest_email = None
    latest_timestamp = 0

    # Loop through the retrieved messages and get the latest one by date
    for msg in messages:
        email_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = email_data["payload"]["headers"]
        timestamp = int(email_data["internalDate"])  # Get email timestamp

        if timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_email = email_data

    if not latest_email:
        print("No valid emails found.")
        return None

    email_headers = latest_email["payload"]["headers"]
    subject = next(header["value"] for header in email_headers if header["name"] == "Subject")
    sender = next(header["value"] for header in email_headers if header["name"] == "From")

    # Extract email body
    email_body = "No text content available."
    if "parts" in latest_email["payload"]:
        for part in latest_email["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                email_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")

    return {
        "subject": subject,
        "from": sender,
        "body": email_body
    }

# Test the function
if __name__ == "__main__":
    email = get_latest_email()
    if email:
        print("\n✅ **Latest Email Details:**")
        print(f"📌 **From:** {email['from']}")
        print(f"📌 **Subject:** {email['subject']}")
        print(f"📌 **Body:**\n{email['body'][:500]}...")

