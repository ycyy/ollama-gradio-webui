import gradio as gr
import ollama

model_list = ollama.list()
model_names = [model['model'] for model in model_list['models']]
def ollama_chat(message, history,model_name):
    stream = ollama.chat(
        model= model_name,
        messages=[
            {
                'role': 'user', 
                'content': message
            }
        ],
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
    text_box = gr.Textbox(scale=4,render=False)
    gr.ChatInterface(fn=ollama_chat,textbox=text_box,additional_inputs=model_info,submit_btn="提交",retry_btn="🔄 重试",undo_btn="↩️ 撤消",clear_btn="🗑️ 清除")

if __name__ == "__main__":
    demo.launch()