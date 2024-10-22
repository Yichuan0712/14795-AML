import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_nth_money_laundering(df, y, n):
    if len(y) != len(df):
        raise ValueError("The length of y must be equal to the number of rows in df.")

    df = df.drop(columns=['Is_laundering'])

    filtered_df = df[y == 1]

    if n < 0 or n >= len(filtered_df):
        raise IndexError(f"Index {n} is out of bounds for the filtered DataFrame.")

    return filtered_df.iloc[n]


def clean_data(input_df):
    alert_message = str({k: (True if v == 1 else v) for k, v in dict(input_df).items() if v != 0})
    return alert_message


def llama_init():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def generate_response(tokenizer, model, device, query, system_prompt_path='reporter/system_prompt.txt', max_length=200):
    system_prompt = f""

    full_prompt = f"{system_prompt}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    return response


def chatbot():
    identity = "the CMU ISO (Carnegie Mellon University Information Security Office) virtual assistant"
    knowledge = """
    1. What should I do if I think my Andrew account has been compromised?
       - You should immediately change your password by going to the Andrew Account page. Contact the CMU Help Center if you need further assistance.

    2. How can I access CMU's VPN?
       - You can download the Cisco AnyConnect VPN from the CMU software catalog and log in using your Andrew ID and password. VPN access ensures your connection is secure while accessing university resources remotely.

    3. How can I recognize phishing emails?
       - Look for suspicious signs such as generic greetings, urgent language, requests for personal information, or links to unrecognized websites. Always report phishing emails to CMU's Information Security Office (iso@cmu.edu).

    4. What should I do if I accidentally clicked on a phishing link?
       - If you clicked on a phishing link, disconnect your device from the internet and contact CMU's Information Security Office (iso@cmu.edu) immediately. Run a virus scan and change any compromised passwords.

    5. How do I securely store sensitive information?
       - Use university-approved encrypted storage solutions such as Box or Google Drive with your Andrew account. Avoid sharing sensitive information through email.

    6. What is Multi-Factor Authentication (MFA) and how do I set it up?
       - MFA adds an extra layer of security by requiring a second verification step. You can set up MFA by going to CMU's Duo portal (www.cmuduo.com) and registering your device for MFA.
    """

    print(f"Welcome to {identity} Chatbot! Type 'exit' to quit.")
    tokenizer, model, device = llama_init()
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        response = generate_response(tokenizer, model, device, query, knowledge)
        print(f"Bot: {response}")

chatbot()