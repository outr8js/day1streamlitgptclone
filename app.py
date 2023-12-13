import gradio as gr
from utils import *

prompt_keys = load_prompts_list_from_json('prompts.json')

with gr.Blocks(gr.themes.Soft(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.purple)) as demo:
    with gr.Row():

        with gr.Column(scale=1, variant = 'panel'):
            # gr.HTML(f"<img src='file/logo.png' width='100' height='100'>")
            files = gr.File(type="filepath", file_count="multiple")
            with gr.Row(equal_height=True):
                vector_index_btn = gr.Button('Create vector store', variant='primary',scale=1)
                vector_index_msg_out = gr.Textbox(show_label=False, lines=1,scale=1, placeholder="Creating vectore store ...")
            
            prompt_dropdown = gr.Dropdown(label="Select a prompt", choices=prompt_keys, value=prompt_keys[0])

            with gr.Accordion(label="Text generation tuning parameters"):
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.1, step=0.05)
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=4096, value=1024, step=1)
                k_context=gr.Slider(label="k_context", minimum=1, maximum=15, value=5, step=1)

            vector_index_btn.click(upload_and_create_vector_store, inputs=[files], outputs=vector_index_msg_out)

        with gr.Column(scale=1, variant = 'panel'):
            with gr.Row(equal_height=True):

                with gr.Column(scale=1):
                    llm = gr.Dropdown(choices= ["gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], 
                                       label="Select the model")

                with gr.Column(scale=1):
                    model_load_btn = gr.Button('Load model', variant='primary',scale=2)
                    load_success_msg = gr.Textbox(show_label=False,lines=1, placeholder="Model loading ...")
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                label='Chatbox', height=725, )

            txt = gr.Textbox(label= "Question",lines=2,placeholder="Enter your question and press shift+enter ")

            with gr.Row():

                with gr.Column(scale=1):
                    submit_btn = gr.Button('Submit',variant='primary', size = 'sm')

                with gr.Column(scale=1):
                    clear_btn = gr.Button('Clear',variant='stop',size = 'sm')

            model_load_btn.click(load_models, [llm], load_success_msg, api_name="load_models")

            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,prompt_dropdown,temperature,max_new_tokens,k_context], chatbot)
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,prompt_dropdown,temperature, max_new_tokens,k_context], chatbot).then(
                    clear_cuda_cache, None, None
                )

            clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    # demo.queue(concurrency_count=3)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
