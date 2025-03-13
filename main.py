from fastapi import FastAPI
from gmail_api import get_latest_email
from fastapi.middleware.cors import CORSMiddleware
from nlp_processing import process_email_content

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Email Processing API Running"}

@app.get("/process-email")
def process_email():
    email = get_latest_email()
    if not email:
        return {"error": "No emails found"}
    
    # Call the saved models via process_email_content
    processed_results = process_email_content(email["body"])

    # Print output in VS Code terminal
    print("\n===== Processed Email Data =====")
    print("Subject:", email["subject"])
    print("From:", email["from"])
    print("Body:", email["body"][:500], "...")  # Truncate long emails
    print("Sentiment:", processed_results["sentiment"])
    print("Summary:", processed_results["summary"])
    print("Suggested Reply:", processed_results["reply"])
    print("================================\n")

    return {
        "subject": email["subject"],
        "from": email["from"],
        "body": email["body"],
        "sentiment": processed_results["sentiment"],
        "summary": processed_results["summary"],
        "reply": processed_results["reply"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)