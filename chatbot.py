import requests

chat_history = []

def ask_ollama(prompt, model="mistral"):
    chat_history.append(f"User: {prompt}")
    full_prompt = "Here's the previous conversation:\n"
    for turn in chat_history[-5:]:
        full_prompt += f"{turn}\n"
    full_prompt += "\nNow answer the current question accordingly."

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False}
        )

        res_json = response.json()
        result = res_json.get("response") or res_json.get("message", {}).get("content", "⚠️ AI returned nothing useful.")
        chat_history.append(f"AI: {result}")
        return result

    except Exception as e:
        return f"⚠️ Error talking to Ollama: {str(e)}"
