from gpt4all import GPT4All
import sys

def stream_reponse(botNum: int, model, prompt: str) -> str:
    print(f"Bot #{botNum}: ", end="", flush=True)
    reponse = ""

    for token in model.generate(prompt, streaming=True):
        reponse += token
        print(token, end="", flush=True)

    print()
    return reponse

def main_loop(models):
    message = "How do I talk to aliens?"

    try:
        while True:
            for num, model in enumerate(models):
                next_message = stream_reponse(num, model, message)
                message = str(next_message)
    except KeyboardInterrupt:
        pass

def with_all_models(models, at):
    if at >= len(models):
        main_loop(models)
        return

    with models[at].chat_session():
        with_all_models(models, at + 1)

def main() -> None:
    device = "NVIDIA GeForce RTX 4060 Laptop GPU"
    device = "cpu"
    models = []

    # Add lots of models and start chat sessions
    for modelNum in range(5):
        models.append(GPT4All(model_name="gpt4all-falcon-newbpe-q4_0.gguf", model_path="/home/nathan/.local/share/nomic.ai/GPT4All", device=device))

    with_all_models(models, 0)

if __name__ == "__main__":
    main()
