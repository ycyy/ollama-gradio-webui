import gradio as gr
import ollama


model_list = ollama.list()
model_names = [model['model'] for model in model_list['models']]
def ollama_chat(message, history,model_name,history_flag):
    messages = []
    message = {
        'role': 'user', 
        'content': message
    }
    if history_flag and len(history)>0:
        for elment in history:
            history_message = {
                'role': 'user', 
                'content': elment[0]
            }
            print(history_message) 
            messages.append(history_message)   
    messages.append(message)
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

with gr.Blocks(title="Ollama Chat", fill_height=True) as demo:
    gr.Markdown("""# Ollama Webui""")
    model_info = gr.Dropdown(model_names, value="", allow_custom_value=True, label="模型", info="选择聊天模型")
    history_flag = gr.Checkbox(label="启用上下文")
    text_box = gr.Textbox(scale=4,render=False)
    gr.ChatInterface(fn=ollama_chat,textbox=text_box,additional_inputs=[model_info,history_flag],submit_btn="提交",retry_btn="🔄 重试",undo_btn="↩️ 撤消",clear_btn="🗑️ 清除")
    

if __name__ == "__main__":
    demo.launch()