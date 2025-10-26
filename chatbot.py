from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime

app = Flask(__name__)

# Load small conversational model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Basic NLP filter (offensive words)
bad_words = ["kill", "hate", "stupid", "suicide", "die"]

def is_offensive(text):
    return any(word in text.lower() for word in bad_words)

# Log user sessions
def log_session(user_msg, bot_reply):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}]\nUser: {user_msg}\nBot: {bot_reply}\n\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    # Check for offensive input
    if is_offensive(user_message):
        bot_reply = "I'm here to listen. It sounds like you're struggling — would you like to talk more about it?"
        log_session(user_message, bot_reply)
        return jsonify({"reply": bot_reply})

    # Generate response
    input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id,temperature=0.7,top_p=0.9,no_repeat_ngram_size=3 )
    bot_reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Simple empathy fallback
    if len(bot_reply.strip()) < 2:
        bot_reply = "I'm here for you. How are you feeling right now?"

    log_session(user_message, bot_reply)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
