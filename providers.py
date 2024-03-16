import os 
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_openai_models(api_key):
    client = OpenAI(api_key=api_key)
    models = client.models.list().model_dump()

    # Define context lengths and prices for specific models
    model_details = {
        'gpt-3.5-turbo-1106': {'context_length': 16385, 'price_per_prompt_1m': 0.001 * 1000, 'price_per_completion_1m': 0.002 * 1000},
        'gpt-3.5-turbo': {'context_length': 4096, 'price_per_prompt_1m': 0.0015 * 1000, 'price_per_completion_1m': 0.002 * 1000},
        'gpt-4-1106-preview': {'context_length': 128000, 'price_per_prompt_1m': 0.01 * 1000, 'price_per_completion_1m': 0.03 * 1000},
        # Add more models and their details as required
    }

    openai_models = {}
    for model in models['data']:
        if model['id'].startswith('gpt'):
            details = model_details.get(model['id'], {})
            model_info = {
                'provider': 'openai',
                'model': model['id']
            }
            model_info.update(details)
            openai_models[model['id']] = model_info

    return openai_models

# Function to get models from Mistral
def get_mistral_models(api_key):
    client = OpenAI(api_key=api_key)
    client.base_url = "https://api.mistral.ai/v1"
    models = client.models.list().model_dump()

    # Define context lengths and prices for specific models
    model_details = {
        'mistral-tiny': {'context_length': 16385, 'price_per_prompt_1m': '0.14 Euro', 'price_per_completion_1m': '0.42 Euro'},
        'mistral-small': {'context_length': 4096, 'price_per_prompt_1m': '0.6 Euro', 'price_per_completion_1m': '1.8 Euro'},
        'mistral-medium': {'context_length': 128000, 'price_per_prompt_1m': '2.5 Euro', 'price_per_completion_1m': '7.5 Euro'},
        # Add more models and their details as required
    }

    mistral_models = {}
    for model in models['data']:
        details = model_details.get(model['id'], {})
        model_info = {
            'provider': 'mistral',
            'model': model['id']
        }
        model_info.update(details)
        mistral_models[model['id']] = model_info

    return mistral_models


def get_openrouter_models(url="https://openrouter.ai/api/v1/models", default_value="openrouter/auto"):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()

            openrouter_models = {}
            for model in data['data']:
                model_info = {
                    'provider': 'openrouter',
                    'description': model['description'],
                    'context_length': model['context_length'],
                    'price_per_prompt_1m': round(float(model['pricing']['prompt']) * 1000000, 2),
                    'price_per_completion_1m': round(float(model['pricing']['completion']) * 1000000, 2)
                }
                openrouter_models[model['id']] = model_info

            return openrouter_models

        else:
            return {default_value: {'provider': 'openrouter', 'description': 'Default model'}}

    except requests.RequestException as e:
        return {default_value: {'provider': 'openrouter', 'description': 'Default model due to error'}}

    
def model_details_to_string(model_details):
    details_str = ", ".join(f"**{key}**: {value}" for key, value in model_details.items())
    return details_str

def init_providers(provider_list=['openrouter']):
    providers = {}
    if 'openai' in provider_list:
        providers['openai'] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1/"
        }
    
    if 'openrouter' in provider_list:
        providers['openrouter'] = {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1"
        }
    
    if 'mistral' in provider_list:
        providers['mistral'] = {
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "base_url": "https://api.mistral.ai/v1/"
        }

    return providers

def init_models(providers):
    all_models = {}

    for provider in providers:
        if provider == 'openrouter':
            openrouter_models = get_openrouter_models()
            all_models['openrouter'] = openrouter_models

        elif provider == 'openai':
            # Custom logic for GPT
            openai_models = get_openai_models(api_key=os.getenv('OPENAI_API_KEY'))
            all_models['openai'] = openai_models

        elif provider == 'mistral':
            mistral_models = get_mistral_models(api_key=os.getenv('MISTRAL_API_KEY'))
            all_models['mistral'] = mistral_models

    return all_models

def get_model_list(provider, all_models):
    return list(all_models[provider].keys())