import gradio as gr
import ollama
import json

model_list = ollama.list()
model_names = [model['model'] for model in model_list['models']]
PROMPT_LIST = []
# 解析 prompt
with open("prompt.json", "r", encoding="utf-8") as f:
    PROMPT_DICT = json.load(f)
    for key in PROMPT_DICT:
        PROMPT_LIST.append(key)
# PROMPT_SELECT = ["翻译助手","周报助手"]
# PROMPT_LIST = {
#     "翻译助手":"你是一个好用的翻译助手。请将我的中文翻译成英文，将非中文的翻译成中文。我发给你所有的话都是需要翻译的内容，你只需要回答翻译结果。翻译结果请符合中文的语言习惯。",
#     "周报助手":"请帮我把以下的工作内容填充为一篇完整的周报，用 markdown 格式以分点叙述的形式输出："
# }
def ollama_chat(message, history,model_name,history_flag):
    messages = []
    chat_message = {
        'role': 'user', 
        'content': message
    }
    if history_flag and len(history)>0:
        for element in history:  
            history_user_message = {
                'role': 'user', 
                'content': element[0]
            }
            history_assistant_message = {
                'role': 'assistant', 
                'content': element[1]
            }
            messages.append(history_user_message)
            messages.append(history_assistant_message)   
    messages.append(chat_message)
    stream = ollama.chat(
        model = model_name,
        messages = messages,
        stream=True
    )
    partial_message = ""
    for chunk in stream:
        if len(chunk['message']['content']) != 0:
            partial_message = partial_message + chunk['message']['content']
            yield partial_message
# 智能体生成
def ollama_prompt(message, history,model_name,prompt_info):
    messages = []
    system_message = {
        'role': 'system', 
        'content': PROMPT_DICT[prompt_info]
    }
    user_message = {
        'role': 'user', 
        'content': message
    }
    messages.append(system_message)
    messages.append(user_message)
    stream = ollama.chat(
        model = model_name,
        messages = messages,       
        stream=True
    )
    partial_message = ""
    for chunk in stream:
        if len(chunk['message']['content']) != 0:
            partial_message = partial_message + chunk['message']['content']
            yield partial_message

with gr.Blocks(title="Ollama WebUI", fill_height=True) as demo:
    with gr.Tab("聊天"):
        with gr.Row():
            with gr.Column(scale=1):
                model_info = gr.Dropdown(model_names, value="", allow_custom_value=True, label="模型选择")
                history_flag = gr.Checkbox(label="启用上下文")
            with gr.Column(scale=4):
                chat_bot = gr.Chatbot(height=600,render=False)
                text_box = gr.Textbox(scale=4,render=False)
                gr.ChatInterface(
                    fn=ollama_chat,
                    chatbot=chat_bot,
                    textbox=text_box,
                    additional_inputs=[model_info,history_flag],
                    submit_btn="提交",
                    retry_btn="🔄 重试",
                    undo_btn="↩️ 撤消",
                    clear_btn="🗑️ 清除",
                    fill_height=True
                )
    with gr.Tab("智能体"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt_model_info = gr.Dropdown(model_names, value="", allow_custom_value=True, label="模型选择")
                prompt_info = gr.Dropdown(choices=PROMPT_LIST,value=PROMPT_LIST[0],label="智能体选择",interactive=True)
            with gr.Column(scale=4):
                prompt_chat_bot = gr.Chatbot(height=540,render=False)
                prompt_text_box = gr.Textbox(scale=4,render=False)
                gr.ChatInterface(
                    fn=ollama_prompt,
                    chatbot=prompt_chat_bot,
                    textbox=prompt_text_box,
                    additional_inputs=[prompt_model_info,prompt_info],
                    submit_btn="提交",
                    retry_btn="🔄 重试",
                    undo_btn="↩️ 撤消",
                    clear_btn="🗑️ 清除",
                    fill_height=True
                )
if __name__ == "__main__":
    demo.launch()