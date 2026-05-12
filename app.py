import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# --- Paths ---
model_folder = "./"  # local folder with tokenizer, adapter, etc.

# --- Load tokenizer and model from local folder ---
tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForCausalLM.from_pretrained(model_folder)

# --- Pipeline ---
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# --- Clean answer to avoid extra Qs ---
def clean_answer(generated_text, question):
    answer = generated_text.replace(f"Q: {question}\nA:", "").strip()
    # Stop at first Q: or double newline
    for stop_token in ["Q:", "\n\n"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0].strip()
    return answer

# --- Ask function ---
def ask(question, max_tokens=200, temperature=0.7):
    prompt = f"Q: {question}\nA:"
    response = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95
    )
    answer = clean_answer(response[0]["generated_text"], question)
    return answer

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SmolLM2 Chatbot\nAsk a question and get a concise answer.")

    # Input box
    question_input = gr.Textbox(
        lines=2,
        placeholder="Type your question here...",
        label="Your Question"
    )

    # Output box (tall and scrollable)
    answer_output = gr.Textbox(
        lines=20,
        placeholder="Answer will appear here...",
        label="Answer",
        interactive=False,
        elem_id="answer_box"
    )

    # Button
    submit_button = gr.Button("Ask")
    submit_button.click(fn=ask, inputs=question_input, outputs=answer_output)

    # Optional CSS for taller output window
    demo.css = """
    #answer_box textarea {
        height: 400px !important;
        font-size: 16px;
        line-height: 1.5;
    }
    """

if __name__ == "__main__":
    demo.launch()