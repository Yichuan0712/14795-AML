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


def load_prompt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return ""


def generate_response(tokenizer, model, device, query, system_prompt_path='reporter/system_prompt.txt', report_template_path='reporter/report_tempalte.txt', max_length=400):
    system_prompt = load_prompt_file(system_prompt_path)
    report_template = load_prompt_file(report_template_path)

    full_prompt = f"{system_prompt}\n\n{report_template}\n\nQuestion: {query}\nAnswer:"

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


def run_risk_assessment_chatbot():
    print("Welcome to the Anti-Money Laundering Risk Assessment Bot. Please type your query below.")
    print("Type 'exit' at any time to quit the session.")

    tokenizer, model, device = llama_init()
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Thank you for using the system. Exiting now.")
            break

        response = generate_response(tokenizer, model, device, query)
        print(f"Bot: {response}")
