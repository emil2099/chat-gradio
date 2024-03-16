import os
import json

import gradio as gr
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import prompts

from providers import init_providers, init_models, model_details_to_string, get_model_list

# Set default configurations
default_providers = ['openrouter', 'openai', 'mistral']
default_provider = 'openrouter'
default_model = 'mistralai/mixtral-8x7b-instruct'
flags = ["Use summary", "Reflect"]
default_flags = []
CONVERSATIONS_FOLDER = 'conversations'
app_port = 7860
custom_js = ""
default_max_tokens_history = 3000

os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)

load_dotenv()
providers = init_providers(default_providers)
models = init_models(providers)
models_list = get_model_list(default_provider, models)
client = OpenAI()

def update_model_dropdown(provider):
    models_for_provider = get_model_list(provider, models)
    return gr.update(choices=models_for_provider, value=models_for_provider[0])

def update_model_details(provider, model):
    return model_details_to_string(models[provider][model])

def set_client(provider, model, client):
    client.base_url = providers[provider]['base_url']
    client.api_key = providers[provider]['api_key']

def load_conversation(conversation_name):
    file_path = f'{CONVERSATIONS_FOLDER}/{conversation_name}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            system_prompt = data['system_prompt']
            history = data['history']
            
            gr.Info(f"Successfully loaded the conversation '{conversation_name}'.")

    else:
        system_prompt = ""
        history = []
        gr.Warning(f"Conversation '{conversation_name}' could not be found.")

    return system_prompt, history, history


def save_conversation(conversation_name, system_prompt, chat_history):
    try:
        if len(chat_history) == 0:
            gr.Warning(f"Can't save empty conversation '{conversation_name}'.")
            return
        
        data = {
            'system_prompt': system_prompt,
            'history': chat_history
        }
        with open(f'{CONVERSATIONS_FOLDER}/{conversation_name}.json', 'w') as file:
            json.dump(data, file)
        gr.Info(f"Conversation '{conversation_name}' has been successfully saved.")
    except Exception as e:
        gr.Warning(f"Failed to save the conversation '{conversation_name}'. Error: {e}")

    conversation_choices = get_conversation_choices()
    return gr.update(choices=conversation_choices)


def delete_conversation(conversation_name):
    file_path = f'{CONVERSATIONS_FOLDER}/{conversation_name}.json'
    if os.path.exists(file_path):
        os.remove(file_path)
        gr.Info(f"Conversation '{conversation_name}' has been successfully deleted.")
    else:
        gr.Warning(f"Could not find the conversation '{conversation_name}' to delete.")

    conversation_choices = get_conversation_choices()
    return gr.update(choices=conversation_choices, value="")


def get_conversation_choices():
    if not os.path.exists(CONVERSATIONS_FOLDER):
        return []

    # Get all .json files
    files = [file for file in os.listdir(CONVERSATIONS_FOLDER) if file.endswith('.json')]

    # Sort the files based on last modification time in descending order
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CONVERSATIONS_FOLDER, x)), reverse=True)

    # Extract the file name without extension
    saved_conversations = [os.path.splitext(file)[0] for file in files]

    return saved_conversations

def update_conversation_dropdown_choices():
    conversation_choices = get_conversation_choices()
    return gr.update(choices=conversation_choices)
    
def count_tokens_per_message(content, encoder="cl100k_base", tokens_per_message = 3):
    encoding = tiktoken.get_encoding(encoder)
    num_tokens = tokens_per_message + len(encoding.encode(content))
    return num_tokens


def calculate_messages_stats(messages):
    total_user_tokens = total_assistant_tokens = total_system_tokens = 0
    total_messages = len(messages)

    for message in messages:
        content = message["content"].rstrip()
        tokens = count_tokens_per_message(content)

        if message["role"] == "user":
            total_user_tokens += tokens
        elif message["role"] == "assistant":
            total_assistant_tokens += tokens
        elif message["role"] == "system":
            total_system_tokens += tokens

    total_tokens = total_user_tokens + total_assistant_tokens + total_system_tokens

    stats_string = (
        f"**Messages**: {total_messages}, **Total Tokens**: {total_tokens}, "
        f"System tokens: {total_system_tokens}, User tokens: {total_user_tokens}, Assistant tokens: {total_assistant_tokens}, "
        f"**Avg Tokens/Msg**: {total_tokens / total_messages if total_messages > 0 else 0:.1f}"
    )

    return stats_string


def update_telemetry(history, system_prompt):
    try:
        messages = prepare_messages(history=history, system_prompt=system_prompt)
        telemetry = calculate_messages_stats(messages)
        return telemetry
    except:
        pass

def get_completion(messages, model, temperature):
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
    return response.choices[0].message.content


def prepare_messages(history, system_prompt=None):
    messages = []
    
    # Add system prompt first if it is not None and not an empty string
    if system_prompt is not None and system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})

    # Add all messages from history
    for human, assistant in history:
        messages.append({"role": "user", "content": human.strip("\n")})
        messages.append({"role": "assistant", "content": assistant.strip("\n")})
    
    return messages

def truncate_messages(messages, token_limit=default_max_tokens_history):
    truncated_history = []
    excluded_history = []
    current_token_count = 0
    system_message_included = False
    token_limit_exceeded = False

    # Handle the first system message
    if messages and messages[0]["role"] == "system":
        system_message = messages[0]
        system_tokens = count_tokens_per_message(system_message["content"])
        current_token_count += system_tokens
        system_message_included = True

    # Iterate over messages in reverse, starting after the system message if present
    start_index = 1 if system_message_included else 0
    for message in reversed(messages[start_index:]):
        if token_limit_exceeded:
            excluded_history.append(message)
            continue

        message_tokens = count_tokens_per_message(message["content"])

        if current_token_count + message_tokens <= token_limit:
            truncated_history.append(message)
            current_token_count += message_tokens
        else:
            # Messages that exceed the token limit are added to excluded_history
            excluded_history.append(message)
            token_limit_exceeded = True

    # Append the system message if it was included
    if system_message_included:
        truncated_history.append(system_message)

    # Reverse the lists to restore the original chronological order
    truncated_history = list(reversed(truncated_history))
    excluded_history = list(reversed(excluded_history)) if excluded_history else None

    return truncated_history, excluded_history

def summarise_conversation(history, model, temperature):
    discarded_messages = True # Set to not none to start the loop
    messages = prepare_messages(history)
    summaries = ""
    while messages:
        truncated_messages, discarded_messages = truncate_messages(messages)

        truncated_messages.append({"role": "user", "content": prompts.summariser})
        summary = get_completion(messages=truncated_messages, model=model, temperature=temperature)
        print(summary)

        summaries += summary + "\n"
        messages = discarded_messages

    return summaries.strip()

def retrieve_response(chatbot_state):
    try:
        return chatbot_state[-1][-1]
    except Exception as e:
        gr.Warning(f"No messages to retrieve.")
        return ""

def replace_response(message, chatbot_state):
    if message == "":
        gr.Warning(f"Cannot replace with empty message. No changes made.")
        return chatbot_state, chatbot_state
    try:
        chatbot_state[-1][-1] = message
        return chatbot_state, chatbot_state
    except Exception as e:
        gr.Warning(f"Failed to replace last response. Error: {e}")
        return chatbot_state, chatbot_state

def predict(message, 
            history,
            provider,
            model, 
            system_prompt,
            temperature,
            max_tokens_response,
            max_tokens_history,
            flags_group,
            summary_text):
    
    set_client(provider, model, client)
    
    if flags_group:
        system_prompt = "System message:\n" + system_prompt    
        if "Use summary" in flags_group and summary_text != "":
            system_prompt = system_prompt + "\n\n" + prompts.use_summary + "\n" + summary_text

    messages = prepare_messages(history, system_prompt)
    messages, _ = truncate_messages(messages, token_limit=max_tokens_history)
    messages.append({"role": "user", "content": message})

    # TODO: Review and test this logic
    if "Reflect" in flags_group:
        print("Reflecting")
        reflect_on = messages.copy()
        reflect_on.append({"role": "user", "content": prompts.reflection})
        reflection = get_completion(reflect_on, model=model, temperature=temperature)
        messages.append({"role": "system", "content": reflection})
    else:
        messages.append({"role": "user", "content": message})

    if "Inject system" in flags_group:
        messages.insert(len(messages), {"role": "system", "content": prompts.system_injection})
    
    print(messages)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens_response
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta is not None:
            delta = chunk.choices[0].delta.content
            if delta:
                partial_message += delta
        yield partial_message

conversation_choices = get_conversation_choices()

with gr.Blocks(js=custom_js) as interface:
    with gr.Accordion(label="Chat settings", open=False, render=False) as chat_accordion:
        provider_dropdown = gr.Dropdown(default_providers, value=default_provider, label="Provider")
        model_dropdown = gr.Dropdown(models_list, value=default_model, label="Model")
        system_prompt = gr.Textbox(label="System prompt", lines=5, placeholder="e.g. You are an unhelpful assistant")
        temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.9, step=0.01, label="Temperature")
        max_tokens_response = gr.Slider(minimum=1, maximum=8092, value=300, step=1, label="Max tokens per response")
        max_tokens_history = gr.Slider(minimum=1, maximum=8092, value=2000, step=1, label="Max tokens history")
        flags_group = gr.CheckboxGroup(flags, value=default_flags, label="Flags")
        model_text = gr.Markdown(value=model_details_to_string(models[default_provider][default_model]))

    with gr.Accordion(label="Manage conversations", open=False, render=False) as conversations_accordion:
        conversation_dropdown = gr.Dropdown(
            choices=conversation_choices, 
            allow_custom_value=True, 
            label="Conversation Name",
            scale=2
        )
        with gr.Row():
            load_button = gr.Button("Load")
            save_button = gr.Button("Save")
            delete_button = gr.Button("Delete")

        with gr.Row():
            telemetry = gr.Markdown(value="No telemetry provided")

    with gr.Accordion(label="Summarise", open=False, render=False) as summarise_accordion:
        summary_button = gr.Button("Summarise")
        summary_text = gr.Textbox(placeholder="Click to summarise...", container=False)

    with gr.Accordion(label="Edit response", open=False, render=False) as edit_response:
        edit_response_textbox = gr.Textbox(placeholder="Retrieve response to edit...", container=False)
        retrieve_response_button = gr.Button("Retrieve")
        replace_response_button = gr.Button("Replace")

    chat = gr.ChatInterface(predict, 
                            additional_inputs_accordion=chat_accordion,
                            additional_inputs=[provider_dropdown, model_dropdown, system_prompt, temperature, 
                                               max_tokens_response, max_tokens_history, flags_group,
                                               summary_text],
                            autofocus=False)
    
    chat_accordion.render()
    conversations_accordion.render()
    summarise_accordion.render()
    edit_response.render()

    load_button.click(fn=load_conversation, inputs=conversation_dropdown, outputs=[system_prompt, chat.chatbot, chat.chatbot_state])
    save_button.click(fn=save_conversation, inputs=[conversation_dropdown, system_prompt, chat.chatbot_state], outputs=[conversation_dropdown])
    delete_button.click(fn=delete_conversation, inputs=conversation_dropdown, outputs=[conversation_dropdown])
    chat.chatbot.change(fn=update_telemetry, inputs=[chat.chatbot_state, system_prompt], outputs=[telemetry])
    summary_button.click(fn=summarise_conversation, inputs=[chat.chatbot_state, model_dropdown, temperature], outputs=[summary_text])
    conversation_dropdown.focus(fn=update_conversation_dropdown_choices, outputs=conversation_dropdown)
    provider_dropdown.change(fn=update_model_dropdown, inputs=[provider_dropdown], outputs=model_dropdown)
    model_dropdown.change(fn=update_model_details, inputs=[provider_dropdown, model_dropdown], outputs=[model_text])
    retrieve_response_button.click(fn=retrieve_response, inputs=[chat.chatbot_state], outputs=[edit_response_textbox])
    replace_response_button.click(fn=replace_response, inputs=[edit_response_textbox, chat.chatbot_state], outputs=[chat.chatbot, chat.chatbot_state])

interface.queue().launch(server_name="0.0.0.0", server_port=app_port)