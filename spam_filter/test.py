import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

test_emails = [
    "Congratulations you won a free iPhone",
    "Are we meeting today?",
    "Claim your reward now",
    "Please find the invoice attached",
    "We have a special offer for you",
    "Let's catch up over lunch tomorrow",
    "Limited time discount on our products",
    "Project deadline is next week",
    "You have been selected for a new job opportunity",
    "Don't forget to submit your report",
    "Use your time wisely and stay productive",
]

X_test = vectorizer.transform(test_emails)
preds = model.predict(X_test)

for email, pred in zip(test_emails, preds):
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"{label} → {email}")
