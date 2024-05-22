from gpt4all import GPT4All
import sys

def main() -> None:
    device = "cpu"
    model1 = GPT4All(model_name="wizardlm-13b-v1.2.Q4_0.gguf", model_path="/home/nathan/.local/share/nomic.ai/GPT4All", device=device)
    model2 = GPT4All(model_name="gpt4all-falcon-newbpe-q4_0.gguf", model_path="/home/nathan/.local/share/nomic.ai/GPT4All", device=device)

    last_reponse = "owo!"

    with open("log.txt", "w") as fp, model1.chat_session(), model2.chat_session():
        while True:
            # Bot one.
            reponse = ""
            print("Bot one: ", end="", flush=True)

            for token in model1.generate(last_reponse, streaming=True, temp=0):
                print(token, end="", flush=True)
                reponse += token

            fp.write(f"Bot one: {reponse}\n")

            # Bot two.
            print("\nBot two: ", end="", flush=True)
            last_reponse = ""

            for token in model2.generate(reponse, streaming=True, temp=0):
                print(token, end="", flush=True)
                last_reponse += token

            fp.write(f"Bot two: {reponse}\n")

            print()

if __name__ == "__main__":
    main()
