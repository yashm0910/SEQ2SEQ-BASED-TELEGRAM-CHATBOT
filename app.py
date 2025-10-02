import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from telegram.ext import Application, MessageHandler, CommandHandler, filters

model_path = "dialogue-model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def reply(user_msg, max_length=60, temperature=0.7, top_p=0.9):
    inputs = tokenizer(
        "chat: " + user_msg,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

API_TOKEN = "8208709847:AAGrYGjCD9-myvRfnbYhbkMpnnc8RhBhIno"

async def start(update, context):
    await update.message.reply_text("Hello! I’m your AI auto-reply bot 🤖")

async def handle_message(update, context):
    user_msg = update.message.text
    bot_reply = reply(user_msg)
    await update.message.reply_text(bot_reply)

def main():
    app = Application.builder().token(API_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()

